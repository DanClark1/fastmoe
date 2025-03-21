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

    Components are defined as a tuple of matricies U, V_1, ..., V_n where U are the global
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

            # print the devices that global_weight and prev_experts.weight are on
            print(global_weight.device, prev_experts.weight.device)
            # stitch global and previous experts
            self.weight = nn.Parameter(torch.cat((global_weight, prev_experts.weight)))
            self.bias = nn.Parameter(torch.cat((global_bias, prev_experts.bias)))

            c_inv = torch.linalg.inv(torch.einsum('kdr,kds->krs', components, components))
            self.c_psuedo_inv = torch.linalg.einsum('kdr,krs,kds->kds', components, c_inv, components)

            # freeze projection matricies
            self.c_psuedo_inv.requires_grad = False


    def forward(self, inp, fwd_expert_count):
        r"""
        Call MOE function
        """
        x = MOELinear.apply(inp, fwd_expert_count, self.weight, self.bias)

        counts = fwd_expert_count if isinstance(fwd_expert_count, torch.Tensor) else \
             torch.tensor(fwd_expert_count, device='cuda')
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

        x_projected = torch.einsum('nkd,kds->nks', padded_outputs, self.c_psuedo_inv)
        return x_projected

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

