from typing import *
from functools import wraps
import bmtrain as bmt
import torch
from .loss_controller import BMPruneLossController
from .strategy import BMPruneStrategy

class BMPrune(bmt.DistributedModule):
    r"""BMPrune adds masks to the model to decrease the model size.

    """
    def __init__(
        self,
        prune_loss_controller: BMPruneLossController,
        strategies: List[BMPruneStrategy]
    ):
        r"""Initialize :class:`BMPrune`.

        Args:
            prune_loss_controller (BMPruneLossController): Controls the way to calculate loss by the masks.
            strategies (List[BMPruneStrategy]): Describes which masks to use.
        """
        super().__init__()
        self.prune_loss_controller = prune_loss_controller
        self.strategies = torch.nn.ModuleList(strategies)

    def set_forward(
        self,
        model,
        forward_fn,
        optimizer
    ):
        r"""Returns a new forward function that includes the pruning loss already.

        Args:
            model (torch.nn.Module): The backbone model.
            forward_fn (callable): The current forward function of trainer.
            optimizer (torch.Optimizer): Optimizer used for updating the learnable parameters of masks.

        Returns:
            Decorated `forward_fn` function.
        """

        strategies = self.strategies
        prune_loss_controller = self.prune_loss_controller

        for strategy in strategies:
            bmt.init_parameters(strategy)
            strategy.set_optimizer(optimizer)
            strategy.inject_mask(model)
            strategy.inject_sparsity(prune_loss_controller.size_calculator)
        
        @wraps(forward_fn)
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
