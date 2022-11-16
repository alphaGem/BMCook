import types
import torch
from torch import Tensor
from torch.nn import Module
from typing import List, Dict, Union, Optional, NewType
from ...cupboard import Cupboard

SPlugin = NewType('Plugin', Dict[str, Optional[Union[int, Tensor]]])

################################# utils for forward-inside pruning setup #################################
def set_pruning_transformer(module: Module, index: int, TRANSFORMER_MASK: List[SPlugin], is_bmtCBlock: bool = True):
    module.index = index
    if is_bmtCBlock:
        module.forward_unprune = module._module.forward
    else:
        module.forward_unprune = module.forward
    def prune_forward(module_self, self_hidden_states, *args, **kwargs):
        index = module_self.index
        mask = TRANSFORMER_MASK[index]['mask']
        out = module_self.forward_unprune(self_hidden_states, *args, **kwargs)
        if mask is not None:
            out = self_hidden_states + (out - self_hidden_states) * mask
        return out
    if is_bmtCBlock:
        module._module.forward = types.MethodType(prune_forward, module)
    else:
        module.forward = types.MethodType(prune_forward, module)

def set_pruning_att(module: Module, index: int, ATTENTION_MASK: List[SPlugin], NUM_HEADS_MASK: List[SPlugin], DIM_HEAD_MASK: List[SPlugin]):
    module.index = index
    module.forward_unprune = module.forward
    def prune_forward(module_self, hidden_states, *args, **kwargs):
        index = module_self.index
        mask = ATTENTION_MASK[index]['mask']
        out = module_self.forward_unprune(hidden_states, *args, **kwargs)
        if mask is not None:
            out = hidden_states + (out - hidden_states) * mask
        return out
    module.forward = types.MethodType(prune_forward, module)
    # prune Linears:
    for s_name, s_module in module.named_modules():
        if 'project' in s_name:
            set_pruning_linear_attention(s_module, index, NUM_HEADS_MASK, DIM_HEAD_MASK, 'in')
        elif 'attention_out' in s_name:
            set_pruning_linear_attention(s_module, index, NUM_HEADS_MASK, DIM_HEAD_MASK, 'out')

def set_pruning_ffn(module: Module, index: int, FFN_MASK: List[SPlugin], DIM_FF_MASK: List[SPlugin]):
    module.index = index
    module.forward_unprune = module.forward
    def prune_forward(module_self, hidden_states, *args, **kwargs):
        index = module_self.index
        mask = FFN_MASK[index]['mask']
        out = module_self.forward_unprune(hidden_states, *args, **kwargs)
        if mask is not None:
            out = hidden_states + (out - hidden_states) * mask
        return out
    module.forward = types.MethodType(prune_forward, module)
    # prune Linears
    for s_name, s_module in module.named_modules():
        if 'w_in.w' in s_name:
            set_pruning_linear_feedforward(s_module, index, DIM_FF_MASK, 'in')
        elif 'w_out' in s_name:
            set_pruning_linear_feedforward(s_module, index, DIM_FF_MASK, 'out')

def set_pruning_linear_attention(module: Module, index: int, NUM_HEADS_MASK: List[SPlugin], DIM_HEAD_MASK: List[SPlugin], in_out: str):
    assert len(NUM_HEADS_MASK) == len(DIM_HEAD_MASK)
    module.index = index
    module.forward_unprune = module.forward
    cb: Cupboard = module.cupboard
    if in_out == 'in':
        multipier1 = cb.dim_in_multipier
        def multipier2():
            return NUM_HEADS_MASK[index]['score']
        cb.dim_in_multipier = lambda: multipier1()*multipier2()
        def prune_forward(module_self, *args, **kwargs):
            index = module_self.index
            num_heads, dim_head = NUM_HEADS_MASK[index]['dim'], DIM_HEAD_MASK[index]['dim']
            num_heads_mask, dim_head_mask = NUM_HEADS_MASK[index]['mask'], DIM_HEAD_MASK[index]['mask']
            out = module_self.forward_unprune(*args, **kwargs)  # (batch, len, num_heads*dim_head)
            if num_heads_mask is not None:
                old_size = out.size()
                out = out.view(old_size[0], old_size[1], num_heads, dim_head)
                out = out * num_heads_mask.view(num_heads, 1)
                out = out.view(old_size[0], old_size[1], num_heads * dim_head)
            if dim_head_mask is not None:
                old_size = out.size()
                out = out.view(old_size[0], old_size[1], num_heads, dim_head)
                out = out * dim_head_mask
                out = out.view(old_size[0], old_size[1], num_heads * dim_head)
            return out
    elif in_out == 'out':
        def prune_forward(module_self, x, *args, **kwargs):
            index = module_self.index
            num_heads, dim_head = NUM_HEADS_MASK[index]['dim'], DIM_HEAD_MASK[index]['dim']
            num_heads_mask, dim_head_mask = NUM_HEADS_MASK[index]['mask'], DIM_HEAD_MASK[index]['mask']
            if num_heads_mask is not None:
                old_size = x.size()  # (batch, len, num_heads * dim_head)
                x = x.view(old_size[0], old_size[1], num_heads, dim_head)
                x = x * num_heads_mask.view(num_heads, 1)
                x = x.view(old_size[0], old_size[1], num_heads * dim_head)
            if dim_head_mask is not None:
                old_size = x.size()
                x = x.view(old_size[0], old_size[1], num_heads, dim_head)
                x = x * dim_head_mask
                x = x.view(old_size[0], old_size[1], num_heads * dim_head)
            out = module_self.forward_unprune(x, *args, **kwargs)  # (batch, len, dim_model)
            return out
    module.forward = types.MethodType(prune_forward, module)

def set_pruning_linear_feedforward(module: Module, index: int, DIM_FF_MASK: List[SPlugin], in_out: str):
    module.index = index
    module.forward_unprune = module.forward
    if in_out == 'in':
        def prune_forward(module_self, *args, **kwargs):
            index = module_self.index
            mask = DIM_FF_MASK[index]['mask']
            out = module_self.forward_unprune(*args, **kwargs)
            if mask is not None:
                out = out * mask
            return out
    elif in_out == 'out':
        def prune_forward(module_self, x, *args, **kwargs):
            index = module_self.index
            mask = DIM_FF_MASK[index]['mask']
            if mask is not None:
                x = x * mask
            out = module_self.forward_unprune(x, *args, **kwargs)
            return out
    module.forward = types.MethodType(prune_forward, module)

