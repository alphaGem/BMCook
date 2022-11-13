import torch
from .cupboard import Cupboard

def inject_cupboard(module:torch.nn.Module):
    module.cupboard = Cupboard(module)
    for child in module.children():
        inject_cupboard(child)