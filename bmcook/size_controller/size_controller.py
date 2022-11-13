import bmtrain as bmt
import torch
from .. import cupboard

class BMSizeController(bmt.DistributedModule):
    r"""Calculates loss according to the model sparsity.
    """
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
    def loss(self):
        return 0

class BrutePenalty(BMSizeController):
    r"""Adds brute penalty calculated by size of the model to the final loss.
    
    The additional loss is :math:`\mathcal{L}=\lambda \times \mathrm{current\_size}/\mathrm{original\_size}`
    """

    def __init__(self, alpha):
        super().__init__(alpha)
    def loss(self):
        return (self.alpha * (self.size_calculator.get_size() / self.original_size)).to(torch.half)

class LagrangianPenalty(BMSizeController):
    def __init__(self, alpha, target_sparsity, lr=-0.01):
        r"""Initializes :class:`Lagrangian Penalty`
        
        The additional loss is calculated as :math:`\mathfrac{L}=\lambda(\lambda_1(s-t)+\lambda_2(s-t)^2)`, 
        where s is the current sparsity and t is the target sparsity

        Args:
            lmbd (float): :math:`\lambda`, which is the coefficiency of the penalty.
            size_calculator: a helper for calculating the model size
            target_sparsity (float): target sparsity
            lr (float): learning rate, which should be **negative** as the lagrangian multipiers 
            should be optimized to maximize the lagrangian term. -0.01 by default.
        """
        super().__init__(alpha)
        if lr > 0:
            raise ValueError("Learning rate of lagrangian multipiers should be negative!")
        self.lr = lr
        self.l1 = bmt.DistributedParameter(
            torch.HalfTensor([0.0]).cuda()
        )
        self.l2 = bmt.DistributedParameter(
            torch.HalfTensor([0.0]).cuda()
        )
        bmt.synchronize()
        self.target_sparsity = target_sparsity
    
    def loss(self, model):
        if self.training:
            cb:cupboard.Cupboard = model.cupboard
            s = (cb.get_expected_size() / cb.get_original_size()).to(torch.half)
            t = self.target_sparsity
            return self.alpha * (self.l1*(s-t) + self.l2*(s-t)*(s-t))
        else:
            return 0

    def get_rate(self, model):
        cb:cupboard.Cupboard = model.cupboard
        bmt.print_rank(cb.get_expected_size(), cb.get_original_size())
        s = (cb.get_expected_size() / cb.get_original_size()).to(torch.half)
        return s
