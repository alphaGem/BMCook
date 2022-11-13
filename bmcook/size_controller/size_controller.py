import bmtrain as bmt
import torch
from .. import cupboard

class BMSizeController(bmt.DistributedModule):
    r"""Calculates loss according to the model sparsity.
    """
    def __init__(self, alpha, size_calculator):
        super().__init__()
        self.alpha = alpha
        self.size_calculator = size_calculator
    def loss(self):
        return 0

class BrutePenalty(BMSizeController):
    r"""Adds brute penalty calculated by size of the model to the final loss.
    
    The additional loss is :math:`\mathcal{L}=\lambda \times \mathrm{current\_size}/\mathrm{original\_size}`
    """

    def __init__(self, alpha, size_calculator):
        super().__init__(alpha, size_calculator)
        self.original_size = self.size_calculator.get_size()
    def loss(self):
        return (self.alpha * (self.size_calculator.get_size() / self.original_size)).to(torch.half)

class LagrangianPenalty(BMSizeController):
    def __init__(self, alpha, size_calculator, target_sparsity, lr=-0.01):
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
        super().__init__(alpha, size_calculator)
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
        cb: cupboard.Cupboard = self.cupboard
        self.original_size = cb.get_original_size()
        bmt.print_rank('model size is', self.original_size)
        self.target_sparsity = target_sparsity
    
    def loss(self):
        if self.training:
            s = (self.size_calculator.get_size() / self.original_size).to(torch.half)
            t = self.target_sparsity
            return self.alpha * (self.l1*(s-t) + self.l2*(s-t)*(s-t))
        else:
            return 0

    def get_rate(self):
        s = (self.size_calculator.get_size() / self.original_size).to(torch.half)
        return s

class LinearSpace:
    def __init__(
        self,
        dim : int
    ):
        self.dim = dim
    
    def get_dim(self):
        return self.dim

class LinearLayer:
    def __init__(
        self,
        space_in : LinearSpace,
        space_out : LinearSpace,
    ):
        self.space_in = space_in
        self.space_out = space_out

    def get_size(self):
        return self.space_in.get_dim()*self.space_out.get_dim()

class Attention:
    def __init__(
        self,
        space_model : LinearSpace,
        num_heads : int,
        dim_head : int,
        shared_kv : bool = False
    ):
        dim_head_kv = 1 if shared_kv else dim_head

        self.space_q = LinearSpace(num_heads * dim_head)
        self.space_k = LinearSpace(num_heads * dim_head_kv)
        self.space_v = LinearSpace(num_heads * dim_head_kv)

        self.proj_q = LinearLayer(space_model, self.space_q)
        self.proj_k = LinearLayer(space_model, self.space_k)
        self.proj_v = LinearLayer(space_model, self.space_v)

    def get_size(self):
        return self.proj_q.get_size()+self.proj_k.get_size()+self.proj_v.get_size()

class FeedForward:
    def __init__(
        self,
        space_model: LinearSpace,
        dim_ff: int
    ):
        self.space_ff = LinearSpace(dim_ff)
        self.w_up = LinearLayer(space_model, self.space_ff)
        self.w_down = LinearLayer(self.space_ff, space_model)

    def get_size(self):
        return self.w_up.get_size()+self.w_down.get_size()



class TransformerBlock:
    def __init__(
        self,
        space_model : LinearSpace,
        dim_ff : int,
        num_heads : int,
        dim_head : int
    ):
        self.attn = Attention(
            space_model,
            num_heads,
            dim_head
        )
        self.ffn = FeedForward(
            space_model,
            dim_ff
        )
    
    def get_size(self):
        return self.attn.get_size()+self.ffn.get_size()


class Encoder:
    def __init__(
        self,
        num_layers : int,
        space_model : LinearSpace,
        dim_ff : int,
        num_heads : int,
        dim_head : int
    ):
        self.layers = [TransformerBlock(
            space_model = space_model,
            dim_ff = dim_ff,
            num_heads=num_heads,
            dim_head=dim_head
        ) for _ in range(num_layers)]

    def get_size(self):
        return sum(layer.get_size() for layer in self.layers)


class GPT2SizeCalculator():
    def __init__(
        self,
        config
    ):
        self.space_model = LinearSpace(config.dim_model)
        self.encoder = Encoder(
            num_layers = config.num_layers,
            space_model = self.space_model,
            dim_ff = config.dim_ff,
            num_heads = config.num_heads,
            dim_head = config.dim_head
        )

    def get_size(self):
        return self.encoder.get_size()