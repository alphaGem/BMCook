import torch
from typing import TypeVar
from model_center.layer.linear import Linear

class Cupboard:
    '''
    A helper class of BMTrain, injected into all the modules of a model.
    '''
    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.has_prune_on_dim_in = False
        self.has_prune_on_dim_out = False
        self.has_prune_sparse = False
    
    def get_original_size(self):
        '''
        Calculates the original size of the module
        '''
        s = 0
        if isinstance(self.module, Linear):
            s = self.module.dim_in * self.module.dim_out
        else:
            for child in self.module.children():
                cc:Cupboard = child.cupboard
                s += cc.get_original_size()
        return s

    def get_expected_size(self):
        '''
        Calculates the expected size of the module
        '''
        s = 0
        if isinstance(self.module, Linear):
            s = self.module.dim_in*self.dim_in_multipier() * self.module.dim_out*self.dim_out_multipier()
        else:
            for child in self.module.children():
                cc:Cupboard = child.cupboard
                s += cc.get_expected_size()
        return s*self.layer_multipier()

    
    def dim_in_multipier(self):
        '''
        Get the multipier on dim_in.
        '''
        return 1.0

    def dim_out_multipier(self):
        '''
        Get the multipier on dim_out.
        '''
        return 1.0

    def layer_multipier(self):
        '''
        Get the multipier of the entire layer.
        '''
        return 1.0

    def sparse_multipier(self):
        return 1.0

    def quant_multipier(self):
        return 1.0

    def generate_protocol(self):
        '''
        Generates the new protocol from the compressing info
        '''
        local_vars = self.module.local_vars
        return local_vars

