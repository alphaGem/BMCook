import json
import torch
import types
import random
import bmtrain as bmt
# from transformers import GPT2ForTokenClassification
# from tqdm import tqdm
import time
import os
import sys

from bmcook.data import MMapIndexedDataset, Dataset
import numpy as np
import pickle as pkl
from model_center.model import GPT2Config, GPT2
import bmcook.pruning as bmp
import bmcook.distilling as bmd
import bmcook

from typing import List

import json

import os
from pathlib import Path


def synchronize_optim_scale(optims: List[torch.optim.Optimizer]):
    r"""
    Synchronize the scale of mutliple optimizers.
    
    Use before `loss = optimizer.loss_scale(loss)` when there are multiple optimizers.

    Args:
        optims (List[torch.optim.Optimizer]): optimizers to be synchronized in scale
        
    Example:
        >>> bmt.synchronize_optimizer_scale([opt1, opt2, opt3])
        >>> loss = opt1.loss_scale(loss) # opt1 can be replaced with opt2 or opt3 because they have the same scale now
        >>> loss.backward()
        >>> bmt.optim_step(opt1, lr1)
        >>> bmt.optim_step(opt2, lr2)
        >>> bmt.optim_step(opt3, lr3)
    """

    min_scale = float('inf')
    for optim in optims:
        if hasattr(optim, 'scale'):
            if optim.scale < min_scale:
                min_scale = optim.scale
        else:
            # no scale, treat as scale = 1
            min_scale = 1

    for optim in optims:
        if hasattr(optim, 'scale') and optim.scale != min_scale:
            optim.justify_scale(min_scale)

def print_inspect(model, name):
    bmt.print_rank(
        bmt.inspect.format_summary(
            bmt.inspect.inspect_model(model, name)
        )
    )

class Trainer:
    @staticmethod
    def batch_iter_shuf(dataset : Dataset, batch_size, rank, world_size):
        st = 0
        end = len(dataset)

        local_size = len(dataset) // world_size
        idx = list(range(local_size))
        random.shuffle(idx)

        batch = []
        while st < local_size:
            it = dataset[idx[st]*world_size + rank]
            if it is not None:
                batch.append( it )
            st += 1
            if len(batch) == batch_size:
                yield {
                    "ctx": torch.stack([it["ctx"] for it in batch]),
                    "len_ctx": torch.LongTensor([it["len_ctx"] for it in batch]),
                    'target': torch.stack([it["target"] for it in batch]),
                }
                batch = []

    @staticmethod
    def batch_iter(dataset : Dataset, batch_size, rank, world_size):
        st = 0
        end = len(dataset)
        batch = []
        while st < end:
            it = dataset[st + rank]
            if it is not None:
                batch.append( it )
            st += world_size
            if len(batch) == batch_size:
                yield {
                    "ctx": torch.stack([it["ctx"] for it in batch]),
                    "len_ctx": torch.LongTensor([it["len_ctx"] for it in batch]),
                    'target': torch.stack([it["target"] for it in batch]),
                }
                batch = []

def main():
    bmt.init_distributed(seed=1)

    gpt_config = GPT2Config.from_pretrained("gpt2-base")
    gpt = GPT2.from_pretrained("gpt2-base", config=gpt_config)
    teacher = GPT2.from_pretrained("gpt2-base", config=gpt_config)
    bmt.synchronize()
    teacher.eval()

    batch_size = 8

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = bmt.optim.AdamOptimizer(gpt.parameters(), lr=1e-5, scale=2**20)
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    
    for _, v in gpt.named_modules():
        if isinstance(v, bmt.TransformerBlockList):

            def new_func(list_self, hidden_states, *args):
                for i in range(len(list_self._modules)):
                    hidden_states = list_self._modules[str(i)](hidden_states, *args)
                return hidden_states

            v.forward = types.MethodType(new_func, v)

            for k in v._modules.keys():
                state_dict = v._modules[k].state_dict()
                for kk, vv in v._modules[k]._module.named_modules():
                    if kk+'.weight' in state_dict:
                        vv.weight.data = state_dict[kk+'.weight'].clone().cuda()
                    if kk+'.bias' in state_dict:
                        vv.bias.data = state_dict[kk+'.bias'].clone().cuda()
                v._modules[k] = v._modules[k]._module

    # NB starts
    # size penalty
    size_loss = bmcook.size_controller.LagrangianPenalty(
        alpha = 1, 
        size_calculator = bmcook.size_controller.GPT2SizeCalculator(gpt_config), 
        target_sparsity = 0.12
    )

    # pruning declaration starts
    MHA_pruning_list = [bmp.strategy.MHALayerPruning(i) for i in range(gpt_config.num_layers)]
    FFN_pruning_list = [bmp.strategy.FFNLayerPruning(i) for i in range(gpt_config.num_layers)]
    MHAs_pruning_list = [bmp.strategy.AttentionHeadPruning(gpt_config.num_heads, i) for  i  in range(gpt_config.num_layers)]
    FFNi_pruning_list = [bmp.strategy.FFNIntermediatePruning(gpt_config.dim_ff, i)  for i in range(gpt_config.num_layers)]
    pruning_list = MHA_pruning_list + FFN_pruning_list + MHAs_pruning_list + FFNi_pruning_list
    pruner = bmp.BMPrune(pruning_list)
    # pruning declaration ends

    # distilling declaration starts
    distill_ce = bmd.strategy.OutputCELoss(
        scale=0.3,
        temp=1
    )

    distill_hidden = bmd.strategy.DynamicHiddenMSELoss(
        scale=0.7,
        projection=True
    )
    distill_hidden.init_w_layer(optimizer,
        dim_t=gpt_config.dim_model,
        dim_s=gpt_config.dim_model,
        projection_init_method='identical')
    for i in range(gpt_config.num_layers):
        distill_hidden.register_layer_student('encoder.layers.'+str(i)+'.ffn.layernorm_before_ffn','post')
    distill_hidden.register_layer_teacher('encoder.layers.2.ffn.layernorm_before_ffn','post')
    distill_hidden.register_layer_teacher('encoder.layers.5.ffn.layernorm_before_ffn','post')
    distill_hidden.register_layer_teacher('encoder.layers.8.ffn.layernorm_before_ffn','post')
    distill_hidden.register_layer_teacher('encoder.layers.11.ffn.layernorm_before_ffn','post')
    distiller = bmd.BMDistill([distill_hidden, distill_ce])
    #distilling declaration ends

    compressor = bmcook.Compressor(gpt, size_loss, pruner, distiller, teacher = teacher)
    #NB ends

    dec_len = 128

    dataset = Dataset(
        MMapIndexedDataset("../openwebtxt/openwebtext_text_document"),
        dec_len
    )

    average_time = 0
    average_time_shift = 0.9

    count = 0
    total_loss = 0.0

    for iteration, data in enumerate(Trainer.batch_iter(dataset, batch_size, bmt.rank(), bmt.world_size())):
        # start training step
        st = time.time()
        dec_input = data["ctx"][:, :dec_len].int()
        dec_length = data["len_ctx"].int()
        dec_mask = torch.arange(dec_len)[None, :].repeat(batch_size, 1) < dec_length[:, None]
        targets = torch.where(dec_mask, data["target"][:, :dec_len].long(), torch.scalar_tensor(-100, dtype=torch.long))

        targets = targets.cuda()
        dec_input = dec_input.cuda()
        dec_length = dec_length.cuda()

        optimizer.zero_grad()
        compressor.zero_grad() # NB
        outputs = gpt(dec_input, dec_length, return_logits=True)[:,:,:-1]
        batch, seq_len, vocab_out_size = outputs.size()
        original_loss = loss_func(outputs.view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))

        original_loss = bmt.sum_loss(original_loss)
        loss = original_loss + compressor.loss() #NB
        global_loss = loss
        synchronize_optim_scale([optimizer, compressor.pruner_optimizer, compressor.size_controller_optimizer]) #NB
        loss = optimizer.loss_scale(loss)

        loss.backward()
        bmt.optim_step(optimizer, lr_scheduler)
        compressor.step() #NB
        iteration_time = time.time() - st
        average_time = average_time * average_time_shift + (1 - average_time_shift) * iteration_time
        bmt.print_rank(global_loss.item(), original_loss.item())
        bmt.print_rank(
            "| Iter: {:6d} | loss: {:.4f} | scale: {:10.4f} | time: {:.4f} | original_loss: {:.4f} |".format(
                iteration,
                global_loss.item(),
                int(optimizer.scale), 
                average_time,
                original_loss.item()
            )
        )


        if iteration % 1000 == 0:
            torch.save(gpt.state_dict(),'results/model.pt')
            torch.save(pruner.state_dict(),'results/masks.pt')
            print_inspect(gpt, "*")

        if iteration % 100 == 0:
            for p in MHA_pruning_list:
                p.print_mask()
            bmt.print_rank('-'*89)
            for p in FFN_pruning_list:
                p.print_mask()
            bmt.print_rank('-'*89)
            for p in MHAs_pruning_list:
                p.print_mask()
            bmt.print_rank('-'*89)
            for p in FFNi_pruning_list:
                p.print_mask()
            bmt.print_rank('-'*89)
            bmt.print_rank(size_loss.get_rate())
            if size_loss.get_rate() < 0.8:
                bmt.print_rank('pruning shut down')
                pruner.eval()
                size_loss.eval()


        


        


if __name__ == '__main__': 
    main()