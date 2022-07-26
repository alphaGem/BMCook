import json
import torch
import types
import random
import bmtrain as bmt
from transformers import GPT2ForTokenClassification
from tqdm import tqdm
import time
import os
import sys

from data import MMapIndexedDataset, Dataset
import numpy as np
import pickle as pkl
from model_center.model import GPT2Config, GPT2
import bmcook.pruning as bmp
import bmcook.distilling as bmd


import json

import os
from pathlib import Path


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

    @staticmethod
    def forward(model, dec_input, dec_length, targets, loss_func):
        outputs = model(
            dec_input, dec_length, return_logits=True)
        logits = outputs
        batch, seq_len, vocab_out_size = logits.size()

        loss = loss_func(logits.view(batch * seq_len, vocab_out_size), targets.view(batch * seq_len))

        #bmt.print_rank(logits)

        return [loss, logits]


def main():
    bmt.init_distributed(seed=1)

    gpt_config = GPT2Config.from_pretrained("gpt2-base")
    gpt = GPT2.from_pretrained("gpt2-base", config=gpt_config)
    teacher = GPT2.from_pretrained("gpt2-base", config=gpt_config)
    bmt.synchronize()

    batch_size = 8

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = bmt.optim.AdamOptimizer(gpt.parameters(), lr=1e-5, scale=2**20)
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

    
    for _, v in gpt.named_modules():
        # bmt.print_rank(_,v)
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


    # pruning declaration starts
    MHA_pruning_list = [bmp.strategy.MHALayerPruning(i) for i in range(gpt_config.num_layers)]
    FFN_pruning_list = [bmp.strategy.FFNLayerPruning(i) for i in range(gpt_config.num_layers)]
    MHAs_pruning_list = [bmp.strategy.AttentionHeadPruning(gpt_config.num_heads, i) for  i  in range(gpt_config.num_layers)]
    FFNi_pruning_list = [bmt.strategy.FFNIntermediatePruning(gpt_config.dim_ff, i)  for i in range(gpt_config.num_layers)]
    pruning_list = MHA_pruning_list + FFN_pruning_list + MHAs_pruning_list + FFNi_pruning_list
    
    prune_loss = bmt.loss_controller.LagrangianPenalty(
        lmbd = 1, 
        size_calculator = bmt.loss_controller.GPT2SizeCalculator(gpt_config), 
        target_sparsity = 0.12, 
        optimizer = optimizer
    )
    pruner = bmp.BMPrune(prune_loss, pruning_list)

    Trainer.forward = pruner.set_forward(
        gpt,
        Trainer.forward,
        optimizer
    )
    bmt.synchronize()
    # pruning declaration ends

    dec_len = 128

    dataset = Dataset(
        MMapIndexedDataset("../openwebtxt/openwebtext_text_document")
    )

    average_time = 0
    average_time_shift = 0.9

    count = 0
    total_loss = 0.0

    prune_end_step = 30000
    distill_end_step = 40000

    for iteration, data in enumerate(Trainer.batch_iter(dataset, batch_size, bmt.rank(), bmt.world_size())):
        
        st = time.time()


        dec_input = data["ctx"][:, :dec_len].int()
        dec_length = data["len_ctx"].int()
        dec_mask = torch.arange(dec_len)[None, :].repeat(batch_size, 1) < dec_length[:, None]
        targets = torch.where(dec_mask, data["target"][:, :dec_len].long(), torch.scalar_tensor(-100, dtype=torch.long))

        targets = targets.cuda()
        dec_input = dec_input.cuda()
        dec_length = dec_length.cuda()

        if iteration <= distill_end_step:
            optimizer.zero_grad()
            outputs = Trainer.forward(
                gpt, dec_input, dec_length, targets, loss_func)

            loss = outputs[0]
            p = outputs[-1]
            global_loss = bmt.sum_loss(loss).item()
            original_loss = bmt.sum_loss(loss-p).item()
            loss = optimizer.loss_scale(loss)

            loss.backward()
            bmt.optim_step(optimizer, lr_scheduler)

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
                bmt.print_rank(prune_loss.get_rate())
                if prune_loss.get_rate() < 0.12 and prune_end_step > iteration + 100:
                    prune_end_step = iteration + 100
                    distill_end_step = iteration + 10100
            
            iteration_time = time.time() - st
            average_time = average_time * average_time_shift + (1 - average_time_shift) * iteration_time
            bmt.print_rank(
                "| Iter: {:6d} | loss: {:.4f} | scale: {:10.4f} | time: {:.4f} | original_loss: {:.4f} |".format(
                    iteration,
                    global_loss,
                    int(optimizer.scale), 
                    average_time,
                    original_loss
                )
            )


        if iteration > distill_end_step:
            outputs = Trainer.forward(
                gpt, dec_input, dec_length, targets, loss_func
            )

            loss = outputs[0]
            global_loss = bmt.sum_loss(loss).item()
            total_loss += global_loss
            count = count +1

            iteration_time = time.time() - st
            average_time = average_time * average_time_shift + (1 - average_time_shift) * iteration_time
            bmt.print_rank(
                "| Iter: {:6d} | loss: {:.4f} | scale: {:10.4f} | time: {:.4f} | avg eval loss: {:.4f}|".format(
                    iteration,
                    global_loss,
                    int(optimizer.scale), 
                    average_time,
                    total_loss/count,
                )
            )

        if iteration == prune_end_step:
            pruner.eval()

            optimizer = bmt.optim.AdamOptimizer(gpt.parameters(), lr=1e-5, scale=2**20)
            lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)

            # distilling declaration starts
            distill_ce = bmd.strategy.OutputCELoss(
                scale=0.3,
                temp=1
            )

            distill_hidden = bmd.strategy.DynamicHiddenMSELoss(
                scale=0.7,
                projection=True
            )
            for i in range(gpt_config.num_layers):
                distill_hidden.register_layer_student('encoder.layers.'+str(i)+'.ffn.layernorm_before_ffn','post')

            distill_hidden.init_w_layer(optimizer,
                dim_t=gpt_config.dim_model,
                dim_s=gpt_config.dim_model,
                projection_init_method='identical')
            
            distill_hidden.register_layer_teacher('encoder.layers.4.ffn.layernorm_before_ffn','post')
            distill_hidden.register_layer_teacher('encoder.layers.13.ffn.layernorm_before_ffn','post')
            distill_hidden.register_layer_teacher('encoder.layers.22.ffn.layernorm_before_ffn','post')
            distill_hidden.register_layer_teacher('encoder.layers.31.ffn.layernorm_before_ffn','post')

            distiller = bmd.BMDistill([distill_hidden, distill_ce])

            Trainer.forward = distiller.set_forward(
                gpt,
                teacher,
                Trainer.forward
            )

            teacher.eval()

            # distilling declaration ends

        if iteration == distill_end_step:
            distiller.eval()
            gpt.eval()

        


        


if __name__ == '__main__': 
    main()