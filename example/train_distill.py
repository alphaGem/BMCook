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
from pruning import BMPrune, m4n2_2d_greedy
from model_center.model import GPT2Config, GPT2
from distilling import BMDistill, HiddenMSELoss
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

    gpt_base_config = GPT2Config.from_pretrained("gpt2-base")
    gpt_large_config = GPT2Config.from_pretrained("gpt2-large")
    gpt = GPT2.from_pretrained("gpt2-base", config=gpt_base_config)
    teacher = GPT2.from_pretrained("gpt2-large", config=gpt_large_config)
    bmt.synchronize()

    batch_size = 8

    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = bmt.optim.AdamOptimizer(gpt.parameters(), scale=2**20)
    lr_scheduler = bmt.lr_scheduler.Noam(optimizer, start_lr=1e-2, warmup_iter=200, end_iter=100000)

    '''
    for k, v in gpt.named_modules():
        bmt.print_rank("=================================================")
        bmt.print_rank(k)
        bmt.print_rank("-------------------------------------------------")
        bmt.print_rank(v)
        bmt.print_rank("=================================================")
    exit()
    '''
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
    
    distill_mse = HiddenMSELoss(
        scale=100,
        projection=True,
        optimizer=optimizer,
        dim_t=1280,
        dim_s=768,
        projection_init_method='from_embedding',
        embedding_t=teacher.input_embedding.weight,
        embedding_s=gpt.input_embedding.weight
    )
    distill_mse.register_layer_pair(
        'encoder.layers.3.ffn.layernorm_before_ffn','post',
        'encoder.layers.3.ffn.layernorm_before_ffn','post'
    )
    distill_mse.register_layer_pair(
        'encoder.layers.6.ffn.layernorm_before_ffn','post',
        'encoder.layers.6.ffn.layernorm_before_ffn','post'
    )
    distill_mse.register_layer_pair(
        'encoder.layers.9.ffn.layernorm_before_ffn','post',
        'encoder.layers.9.ffn.layernorm_before_ffn','post'
    )
    Trainer.forward = BMDistill.set_forward(
        gpt,
        teacher,
        Trainer.forward,
        [distill_mse]
    )

    teacher.eval()

    bmt.synchronize()

    # dec_len = 128
    dec_len = 512

    dataset = Dataset(
        MMapIndexedDataset("../openwebtxt/openwebtext_text_document")
    )

    average_time = 0
    average_time_shift = 0.9

    for iteration, data in enumerate(Trainer.batch_iter(dataset, batch_size, bmt.rank(), bmt.world_size())):
        st = time.time()

        optimizer.zero_grad()

        dec_input = data["ctx"].int()
        dec_length = data["len_ctx"].int()
        dec_mask = torch.arange(dec_len)[None, :].repeat(batch_size, 1) < dec_length[:, None]
        targets = torch.where(dec_mask, data["target"].long(), torch.scalar_tensor(-100, dtype=torch.long))

        targets = targets.cuda()
        dec_input = dec_input.cuda()
        dec_length = dec_length.cuda()

        outputs = Trainer.forward(
            gpt, dec_input, dec_length, targets, loss_func)

        loss = outputs[0]
        global_loss = bmt.sum_loss(loss).item()
        loss = optimizer.loss_scale(loss)

        loss.backward()
        bmt.optim_step(optimizer, lr_scheduler)

        if iteration % 1000 == 0:
            print_inspect(gpt, "*")

        # logging
        iteration_time = time.time() - st
        average_time = average_time * average_time_shift + (1 - average_time_shift) * iteration_time

        bmt.print_rank(
            "| Iter: {:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | time: {:.4f}".format(
                iteration,
                global_loss,
                lr_scheduler.current_lr, 
                int(optimizer.scale), 
                average_time / (1 - pow(average_time_shift, iteration + 1))
            )
        ) 


if __name__ == '__main__': 
    main()