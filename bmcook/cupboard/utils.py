import torch
from .cupboard import Cupboard

def inject_cupboard(module:torch.Module):
    module.cupboard = Cupboard(module)
    for child in module.named_children:
        inject_cupboard(child)