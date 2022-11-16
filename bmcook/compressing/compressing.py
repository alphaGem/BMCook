import bmtrain as bmt
import torch
from .. import cupboard
from ..pruning import BMPrune
from .cost_penalty import CostPenalty
from typing import Dict, Type

class BMCompress:
    def __init__(self, model, doSprune: bool, pruner: Type[BMPrune], penalty: CostPenalty):
        self.model = model
        self.doSprune = doSprune
        self.pruner = pruner
        self.penalty = penalty

    def set_forward(self, forward_fn):
        bmt.print_rank('bmcompress set forward')
        cupboard.utils.inject_cupboard(self.model)
        bmt.print_rank('injected')
        penalty = self.penalty
        def forward(model, loss_func, targets, *model_args, **model_kwargs):
            outputs = forward_fn(model, loss_func, targets, *model_args, **model_kwargs)
            loss = outputs[0]

            if self.doSprune:
                self.pruner.sprune_engine.update()
            
            lag_loss = penalty.loss()
            sparsity = penalty.get_sparsity()

            loss += lag_loss
            
            outputs[0], outputs[2], outputs[3] = loss, lag_loss, sparsity
            return outputs
        bmt.print_rank('got forward')
        return forward

    def step(self):
        if self.penalty.has_optimizer:
            self.penalty.optimizer.step()
        self.pruner.sprune_engine.sp_optimizer.step()

    def zero_grad(self):
        if self.penalty.has_optimizer:
            self.penalty.optimizer.zero_grad()
        self.pruner.sprune_engine.sp_optimizer.zero_grad()
