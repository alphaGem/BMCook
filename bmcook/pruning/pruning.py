from typing import List
import bmtrain as bmt
import torch
from .BMPruneLoss import BMPruneLossController
from .BMPruneStrategy import BMPruneStrategy

class BMPrune(bmt.DistributedModule):
    def __init__(
        self,
        prune_loss_controller : BMPruneLossController,
        strategies: List[BMPruneStrategy]
    ):
        super().__init__()
        self.prune_loss_controller = prune_loss_controller
        self.strategies = torch.nn.ModuleList(strategies)

    def set_forward(
        self,
        model,
        forward_fn,
        optimizer):

        strategies = self.strategies
        prune_loss_controller = self.prune_loss_controller

        for strategy in strategies:
            bmt.init_parameters(strategy)
            strategy.set_optimizer(optimizer)
            strategy.inject_mask(model)
            strategy.inject_sparsity(prune_loss_controller.size_calculator)
        
        def forward(model, dec_input, dec_length, targets, loss_func):
            outputs = forward_fn(
                model, dec_input, dec_length, targets, loss_func
            )


            if self.training:
                loss = outputs[0]
                p_loss = prune_loss_controller.get_loss()
                loss = loss + p_loss

                outputs[0] = loss
                outputs = outputs + [p_loss, ]


            return outputs

        return forward
