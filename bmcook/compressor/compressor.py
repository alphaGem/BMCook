import torch
import bmtrain as bmt
from .. import pruning as bmp
from .. import distilling as bmd
from .. import quant as bmq
from .. import moe as bme
from .. import size_controller as bmsize
from .. import cupboard

class Compressor(bmt.DistributedModule):
    def __init__(
        self,
        model,
        size_controller: bmsize.BMSizeController,
        pruner: bmp.BMPrune = None,
        distiller: bmd.BMDistill = None,
        quantizer: bmq.BMQuant = None, # TODO to be supported
        moefier: bme.BMMoE = None, # TODO to be supported
        teacher: torch.nn.Module = None
    ):
        super().__init__()
        self.model = model
        self.pruner = pruner
        self.distiller = distiller
        self.quantizer = quantizer
        self.moefier = moefier
        self.size_controller = size_controller

        cupboard.utils.inject_cupboard(model)

        if pruner: pruner.set_forward(model, size_controller)
        if distiller: distiller.set_forward(model, teacher)

        if any(param.requires_grad for param in pruner.parameters()):
            self.pruner_optimizer = bmt.optim.AdamOptimizer([
                {
                    'params': pruner.parameters(),
                    'lr': 0.01
                }
            ], scale=2**20)
        if any(param.requires_grad for param in size_controller.parameters()):
            self.size_controller_optimizer = bmt.optim.AdamOptimizer([
                {
                    'params': size_controller.parameters(),
                    'lr': size_controller.lr
                }
            ], scale=2**20)

    def zero_grad(self):
        if self.pruner_optimizer: self.pruner_optimizer.zero_grad()
        if self.size_controller_optimizer: self.size_controller_optimizer.zero_grad()

    def step(self):
        if self.pruner_optimizer and self.pruner.training:
            bmt.optim_step(self.pruner_optimizer)
        if self.size_controller_optimizer and self.size_controller.training:
            bmt.optim_step(self.size_controller_optimizer)

    def loss(self):
        loss = self.distiller.loss() + self.size_controller.loss(self.model)
        return loss

    
        