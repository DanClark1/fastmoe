r"""
FMoE's parallel linear layer
"""
import torch
import torch.nn as nn
from torch.autograd import Function
import math

import fmoe_cuda
from fmoe.linear import MOELinear, FMoELinear


class FMoELinearProj(nn.Module):
    r"""
    A linear layer that contains multiple experts, projected onto defined components.
    Takes in previously trained experts and projects them onto the components. Will
    also create a new global expert.
    As multiple experts can be placed on the same worker, the computation can be
    performed in parallel to increase the performance.
    The FMoELinear module provides such function.

    Components are defined as a tensor of matricies U, V_1, ..., V_n where U are the global
    components and V_1, ..., V_n are the local components. The global expert is the first
    expert in the list of experts.
    """

    def __init__(
        self,
        components,
        num_expert: int = 0,
        in_feat: int = 0,
        out_feat: int = 0,
        bias: bool = True,
        rank: int = 0,
        prev_experts: FMoELinear = None,
    ):
        
        super().__init__()
        if prev_experts is None:
            self.num_expert = num_expert
            self.in_feat = in_feat
            self.out_feat = out_feat
            self.rank = rank
            self.weight = nn.Parameter(torch.Tensor(num_expert, out_feat, in_feat))
            if bias:
                self.bias = nn.Parameter(torch.zeros(num_expert, out_feat))
            else:
                self.register_parameter("bias", None)

            self.reset_parameters()
        else:
            self.num_expert = len(prev_experts.weight) + 1
            self.in_feat = prev_experts.in_feat
            self.out_feat = prev_experts.out_feat
            self.rank = prev_experts.rank
            global_weight = torch.zeros(1, self.out_feat, self.in_feat, device=prev_experts.weight.device)
            global_bias = torch.zeros(1, self.out_feat, device=prev_experts.bias.device)

            # initialise
            torch.nn.init.kaiming_uniform_(global_weight, a=math.sqrt(5))
            self.weight = nn.Parameter(global_weight)

            # stitch global and previous experts
            self.weight = nn.Parameter(torch.cat((global_weight, prev_experts.weight)))
            self.bias = nn.Parameter(torch.cat((global_bias, prev_experts.bias)))

            self.components = components
            # freeze projection matricies
            self.components.requires_grad = False

            # Add trainable upscaling matrices to project back to original dimension
            # Components shape is expected to be (n_experts, source_dim, target_dim)
            if components is not None:
                target_dim = components.size(-2)
                source_dim = components.size(-1)
                self.upscale_proj = nn.Parameter(
                    torch.zeros(self.num_expert, target_dim, source_dim, device='cuda') * 0.02,
                )
                torch.nn.init.kaiming_uniform_(self.upscale_proj, a=math.sqrt(5))
                if bias:
                    self.upscale_bias = nn.Parameter(torch.zeros(self.num_expert, source_dim, device='cuda'))
                else:
                    self.register_parameter("upscale_bias", None)


    def forward(self, inp, fwd_expert_count):
        r"""
        Call MOE function
        """

        print(inp.shape, fwd_expert_count)

        inp_flat = inp.view(-1, inp.size(-1))
        # sanity checks
        assert inp_flat.ndim == 2, "Input must be 2‑D"
        assert fwd_expert_count.ndim == 1, "fwd_expert_count must be 1‑D"
        assert fwd_expert_count.shape[0] == self.num_expert, (
            f"Expected {self.num_expert} experts, got {fwd_expert_count.shape[0]}"
        )
        assert fwd_expert_count.sum().item() == inp_flat.shape[0], (
            f"Sum of counts ({fwd_expert_count.sum().item()}) != rows ({inp_flat.shape[0]})"
        )
        x = MOELinear.apply(inp.type_as(self.weight), fwd_expert_count, self.weight, self.bias)

        x_projected = self.project(x, fwd_expert_count, inp)

        return x_projected
    

    def project(self, x, fwd_expert_count, inp):
        r"""
        Project the output of the experts onto the components
        """
        # ensure counts is a Tensor on the same device as inp
        if isinstance(fwd_expert_count, torch.Tensor):
            counts = fwd_expert_count.to(inp.device)
        else:
            counts = torch.tensor(fwd_expert_count, device=inp.device)

        n_experts = counts.shape[0]
        total_tokens = x.shape[0]
        max_tokens = counts.max().item()

        expert_ids = torch.arange(n_experts, device='cuda').repeat_interleave(counts)


        offsets = torch.cat([torch.tensor([0], device='cuda'), counts.cumsum(dim=0)])
        token_indices = torch.arange(total_tokens, device='cuda')
        token_positions = token_indices - offsets[expert_ids]
        
        padded_outputs = torch.zeros(n_experts, max_tokens, x.shape[1], device='cuda', dtype=x.dtype)
        
        padded_outputs[expert_ids, token_positions] = x

        # getting them in shape (max_tokens (effetively n_tokens), n_experts, n_features)
        padded_outputs = padded_outputs.transpose(0, 1)

        x_projected = torch.einsum('nkd,ksd->nks', padded_outputs, self.components)


        # Project back up to original dimension using trainable matrices
        x_projected_upscaled = torch.einsum('nks,ksd->nkd', x_projected, self.upscale_proj)
        
        # Add bias if present
        if self.upscale_bias is not None:
            # Add the bias to each token (broadcasting across the max_tokens dimension)
            x_projected_upscaled = x_projected_upscaled + self.upscale_bias.unsqueeze(0)


        # Remove padding by transposing and using advanced indexing
        # Transpose to [n_experts, max_tokens, feature_dim]
        x_projected_t = x_projected.transpose(0, 1)
        
        # Use advanced indexing to gather the correct elements
        # For each token, get its projected value from the right expert and position
        output = x_projected_t[expert_ids, token_positions]
        
        # Reshape back to match input shape
        original_shape = list(inp.shape)
        original_shape[-1] = output.size(-1)
        output = output.view(original_shape)

        return output



    def extra_repr(self) -> str:
        return "num_expert={}, in_features={}, \
        out_features={}, bias={}, rank={}".format(
            self.num_expert,
            self.in_feat,
            self.out_feat,
            self.bias is not None,
            self.rank,
        )

    def reset_parameters(self):
        # Approach is the same as in torch.nn.Linear
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L88
        # bias is left to zero, similar as megatron

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

