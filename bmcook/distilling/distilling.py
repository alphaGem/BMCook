import types
from typing import List
from functools import wraps
import torch
import bmtrain as bmt
import torch.nn.functional as F
import model_center
from .strategy import BMDistillStrategy


def inject_inspection(model, modules, suffix):
    def generate_new_forward(f, t, inspect_name):
        if  t == 'pre':
            @wraps(f)
            def _forward(x, **kwargs):
                bmt.inspect.record_tensor(x, inspect_name)
                return f(x, **kwargs)
        
        elif modules[k]['type'] == 'post':
            @wraps(f)
            def _forward(x, **kwargs):
                x = f(x, **kwargs)
                bmt.inspect.record_tensor(x, inspect_name)
                return x
        return _forward

    select_keys = set()
    for k, v in model.named_modules():
        if k in modules:
            select_keys.add(k)
            v.forward = generate_new_forward(v.forward, modules[k]['type'], k+suffix)
    
    bmt.print_rank('Selected modules for hidden state MSE: {}'.format(select_keys))   

def read_inspection(records, modules, suffix, detach=False):
    result = {}
    for k in modules:
        if detach:
            result[k] = records[k+suffix].detach()
        else:
            result[k] = records[k+suffix]
    return result

class BMDistill(bmt.DistributedModule):
    '''
    BMDistill provide additional training objectives for knowledge distillation, which further improves the performance of compressed models.
    '''
    def __init__(self, strategies: List[BMDistillStrategy]):
        r"""Initializes BMDistill class.
        
        Args:
            strategies (List[BMDistillStrategy]): Strategies to be used in the distilling process. 
        """
        super().__init__()
        self.strategies = strategies
        self.module_s = {}
        self.module_t = {}
        for strategy in strategies:
            self.module_s.update(strategy.module_s)
            self.module_t.update(strategy.module_t)

    def set_forward(self,
        student: torch.nn.Module,
        teacher):
        '''
        Modify the forward function of the student model to compute additional knowledge distillation loss.
        `student` and `teacher` should return (logits, hidden_states, att_scores).
        logits: (batch_size, vocab_size)
        hidden_states: (batch_size, dec_len, hidden_size)
        att_scores: (batch_size, dec_len, enc_len)

        Args:
            student (torch.nn.Module): Student model.
            teacher (torch.nn.Module): Teacher model.
            forward_fn (callable): Forward function of the student model.

        Returns:
            Decorated `forward_fn` function.
        '''
        # Inspect all necessary tensors
        inject_inspection(student, self.module_s, '_student')
        inject_inspection(teacher, self.module_t, '_teacher')

        set_loss = self.set_loss
        training = self.training

        def decorate_forward(f):
            @wraps(f)
            def new_forward(dec_input, dec_length, *args, **kwargs):
                # Get the output of both student and teacher models.
                with bmt.inspect.inspect_tensor() as inspector:
                    outputs = f(
                        dec_input, dec_length, *args, **kwargs
                    )
                    outputs_t = teacher(
                        dec_input, dec_length, return_logits = True
                    )
                # Get necessary tensors of hidden layers
                records = {}
                for record in inspector._summary:
                    records[record['name']] = record['tensor']
                records_s = read_inspection(records, self.module_s, '_student')
                records_t = read_inspection(records, self.module_t, '_teacher', detach=True)
                logits_s = outputs
                logits_t = outputs_t.detach()
                
                # Calculation of modified loss
                d_loss = 0
                if training:
                    for strategy in self.strategies:
                        d_loss = d_loss + strategy.loss(logits_s, logits_t, records_s, records_t)

                set_loss(d_loss)

                return outputs
            return new_forward

        student.forward = decorate_forward(student.forward)

    def set_loss(self, loss):
        self.d_loss = loss
    def loss(self):
        return self.d_loss