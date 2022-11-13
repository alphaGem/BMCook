from typing import *
from functools import wraps
import bmtrain as bmt
import torch
from .strategy import BMPruneStrategy
from .. import size_controller

class BMPrune(bmt.DistributedModule):
    r"""BMPrune adds masks to the model to decrease the model size.

    """
    def __init__(
        self,
        strategies: List[BMPruneStrategy]
    ):
        r"""Initialize :class:`BMPrune`.

        Args:
            prune_loss_controller (BMPruneLossController): Controls the way to calculate loss by the masks.
            strategies (List[BMPruneStrategy]): Describes which masks to use.
        """
        super().__init__()
        self.strategies: List[BMPruneStrategy] = torch.nn.ModuleList(strategies)

    def set_forward(
        self,
        model: torch.nn.Module,
        size_con: size_controller.BMSizeController
    ):
        r"""Returns a new forward function that includes the pruning loss already.

        Args:
            model (torch.nn.Module): The backbone model.
            forward_fn (callable): The current forward function of trainer.

        Returns:
            None
        """

        strategies = self.strategies

        for strategy in strategies:
            bmt.init_parameters(strategy)
            strategy.inject_mask(model)
