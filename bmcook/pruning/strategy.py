from abc import abstractmethod
from time import sleep
from turtle import forward
import types
import torch
import bmtrain as bmt
import torch.nn.functional as F
from torch.autograd import Variable
from model_center.layer import Attention
from ..cupboard import Cupboard

limit_a, limit_b, epsilon = -.1, 1.1, 1e-4

class BMPruneStrategy(bmt.DistributedModule):
    r"""Abstract class for pruning strategies.

    """
    def __init__(self,
        targets,
        type,
        adjacant=[]):
        r"""Initializes :class:`BMPruneStrategy`.

        Adds masks to layers.
        
        Let the layer be :math:`f: \mathbb{R}^n \to \mathbb{R}^m; x \mapsto f(x)`, and let the mask be :math:`z`.

        If `type` is set to 'pre', we have :math:`z \in \mathbb{R}^n`,
        and the new linear layer is :math:`x \mapsto f(x \cdot \mathrm{diag}(z))`.

        If `type` is set to 'post', we have :math:`z \in \mathbb{R}^m`,
        and the new linear layer is :math:`x \mapsto f(x) \cdot \mathrm{diag}(z)`.

        `type=layerwise` is for pruning on an entire layer.

        If `type` is set to 'layerwise', we have :math:`z \in \mathbb{R}`,
        and the new layer is :math:`x \mapsto zf(x)`

        Args:
            targets:
                List of Linear(), targets to be pruned.
            type:
                Choose from 'pre' or 'post': whether to add the mask before or after the layer.
        """
        super().__init__()
        self.targets = targets
        self.prune_type = type
        self.adjacant = adjacant
    
    def set_optimizer(self, optimizer):
        optimizer.add_param_group({'params': self.parameters(), 'lr': 0.01})

    @abstractmethod
    def get_mask(self):
        pass

    @abstractmethod
    def get_sparsity(self):
        pass

    def print_targets(self):
        bmt.print_rank(self.targets)

    @abstractmethod
    def apply_mask(self, x):
        pass

    def inject_mask(self, model):
        for k, v in model.named_modules():
            if k in self.targets:
                cb:Cupboard = v.cupboard
                f = v.forward
                if self.prune_type == 'pre':
                    g = cb.dim_in_multipier
                    cb.dim_in_multipier = lambda: g()*self.get_sparsity()
                    def _forward(x, **kwargs):
                        x = self.apply_mask(x)
                        return f(x, **kwargs)
                elif self.prune_type == 'post':
                    g = cb.dim_out_multipier
                    cb.dim_out_multipier = lambda: g()*self.get_sparsity()
                    def _forward(*input, **kwargs):
                        x = f(*input, **kwargs)
                        return self.apply_mask(x)
                elif self.prune_type == 'layerwise':
                    g = cb.layer_multipier
                    cb.layer_multipier = lambda: g()*self.get_sparsity()
                    def _forward(*input, **kwargs):
                        x = f(*input, **kwargs)
                        return self.apply_mask(x)
                v.forward = _forward
            
            if k in self.adjacant:
                cb:Cupboard = v.cupboard
                if self.prune_type == 'pre':
                    g = cb.dim_out_multipier
                    cb.dim_out_multipier = lambda: g()*self.get_sparsity()
                elif self.prune_type == 'post':
                    g = cb.dim_in_multipier
                    cb.dim_in_multipier = lambda: g()*self.get_sparsity()


class HardConcretePruning(BMPruneStrategy):
    def __init__(self, dim, targets, type, adjacant = []):
        self.dim = dim
        
        super().__init__(
            targets = targets,
            type = type,
            adjacant = adjacant
        )
        self.loga =  bmt.DistributedParameter(
            torch.HalfTensor([2.5]*dim)
        )
        bmt.synchronize()

    def quantile_concrete(self, x, loga):
        x = x.to(torch.float)
        loga = loga.to(torch.float)
        y = torch.sigmoid((torch.log(x) - torch.log(1-x) + loga))
        return y * (limit_b - limit_a) + limit_a

    def get_eps(self, size):
        if self.training:
            eps = torch.HalfTensor(size).uniform_(epsilon, 1-epsilon).cuda()
        else:
            eps = torch.HalfTensor([0.5]).cuda().broadcast_to(size)
        return eps

    def get_mask(self):
        z = self.quantile_concrete(self.get_eps(self.loga.size()), self.loga)
        z = F.hardtanh(z, min_val=0, max_val=1)
        self.z_tmp = z
        z = z.to(torch.half)
        return z

    def apply_mask(self, x):
        z = self.get_mask().to(x.device)
        x = x * z
        return x

    def get_sparsity(self):
        shift = torch.HalfTensor([2.4]).cuda()
        loga = self.loga.to(torch.float)
        shift = shift.to(torch.float)
        s = torch.sigmoid(loga+shift).mean()
        return s

    def print_mask(self):
        avg = torch.HalfTensor([0.5]).cuda()
        bmt.print_rank(self.quantile_concrete(avg, self.loga).mean())

    def is_mask_zero(self): # return value may vary when self.training=True
        z = self.get_mask()
        return torch.le(z, 0).all()





class MHALayerPruning(HardConcretePruning):
    def __init__(self, layer):
        self.layer = layer
        super().__init__(
            dim = 1,
            targets = ['encoder.layers.'+str(layer)+'.self_att.self_attention'],
            type = 'layerwise'
        )




class FFNLayerPruning(HardConcretePruning):
    def __init__(self, layer):
        self.layer = layer
        super().__init__(
            dim = 1,
            targets = ['encoder.layers.'+str(layer)+'.ffn.ffn'],
            type = 'layerwise'
        )

class AttentionHeadPruning(HardConcretePruning):
    def __init__(self, num_heads, layer):
        self.layer = layer
        self.num_heads = num_heads
        super().__init__(
            dim = num_heads,
            targets = ['encoder.layers.'+str(layer)+'.self_att.self_attention.attention_out'],
            type = 'pre',
            adjacant = ['encoder.layers.'+str(layer)+'.self_att.self_attention.project_q']+
                       ['encoder.layers.'+str(layer)+'.self_att.self_attention.project_k']+
                       ['encoder.layers.'+str(layer)+'.self_att.self_attention.project_v']
        )


    def apply_mask(self, x):
        '''
        :param x: (batch_size, dim_model, num_heads * dim_head)
        '''
        z = self.get_mask().to(x.device)
        batch_size, dim_model, dim_heads = x.size()
        num_heads = self.num_heads
        dim_head = dim_heads//num_heads
        x = x.view(batch_size, dim_model, num_heads, dim_head)
        x = x * z[:, None]
        x = x.view(batch_size, dim_model, dim_heads)
        return x

    


class FFNIntermediatePruning(HardConcretePruning): 
    def __init__(self, dim_int, layer):
        self.dim_int = dim_int
        self.layer = layer
        super().__init__(
            dim = dim_int,
            targets = ['encoder.layers.'+str(layer)+'.ffn.ffn.w_in'],
            type = 'post',
            adjacant = ['encoder.layers.'+str(layer)+'.ffn.ffn.w_out']
        )

    