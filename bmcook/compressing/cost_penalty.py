import bmtrain as bmt
import torch
from .. import cupboard
from typing import Dict

class CostPenalty(torch.nn.Module):
    r"""Calculates loss according to the model size.

    Args:
            model: `torch.nn.Module`, the model that the size controller is working on
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.has_optimizer = False

    def get_sparsity(self):
        cb:cupboard.Cupboard = self.model.cupboard
        # bmt.print_rank(cb.get_expected_size(), cb.get_original_size())
        s = (cb.get_expected_size() / cb.get_original_size()).to(torch.half)
        return s

    def loss(self):
        return 0

class BruteSparsityPenalty(CostPenalty):
    r"""Adds brute penalty calculated by size of the model to the final loss.
    
    The additional loss is :math:`\mathcal{L}=\lambda \times \mathrm{current\_size}/\mathrm{original\_size}`
    """

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
    def loss(self):
        return self.alpha * self.get_sparsity()

class LagrangianSparsityPenalty(CostPenalty):
    r"""Lagrangian sparsity penalty.
    
    The additional loss is calculated as :math:`\mathfrac{L}=\lambda(\lambda_1(s-t)+\lambda_2(s-t)^2)`, 
    where s is the current sparsity and t is the target sparsity
    
    """
    def __init__(self,
            model:torch.nn.Module,
            criterion:str,
            target_mode:str,
            target_sparsity:float,
            lr:float=-0.1
        ):
        r"""The constructor of class :class:`LagrangianSparsityPenalty`

        Args:
            model: `torch.nn.Module`, the model that the size controller is working on.
            criterion: `str`, only `l0` is supported now.
            target_mode: `str`, choose from `sparsity` and `dimension`.
            target_sparsity: `float`, target sparsity.
            lr: `float`, learning rate, which should be **negative** as the lagrangian multipiers 
            should be optimized to maximize the lagrangian term. -0.1 by default.
        """
        super().__init__(model)

        self.criterion = criterion
        assert self.criterion == 'l0', "BMCook sparsity controller do not support other criterions besides l0 yet."
        self.target_mode = target_mode
        self.target_sparsity = target_sparsity

        if lr > 0:
            raise ValueError("Learning rate of lagrangian multipiers should be negative!")
        self.lr = lr

        self.lambda_1 = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float, device='cuda'))
        self.lambda_2 = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float, device='cuda'))

        lagrangian_params = [{
                    "params": [self.lambda_1, self.lambda_2],
                    "weight_decay": 0.0,
                    "lr": lr
                }]
        self.optimizer = torch.optim.AdamW(lagrangian_params)
        self.has_optimizer = True
    
    def loss(self):
        if self.training:
            s = self.get_sparsity()
            t = self.target_sparsity
            return self.lambda_1*(s-t) + self.lambda_2*(s-t)*(s-t)
        else:
            return 0
