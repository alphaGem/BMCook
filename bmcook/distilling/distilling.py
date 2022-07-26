import types
from typing import List
import bmtrain as bmt
import torch.nn.functional as F
import model_center
from .BMDistillStrategy import BMDistillStrategy


def inject_inspection(model, modules, suffix):
    def generate_new_forward(f, t, inspect_name):
        if  t == 'pre':
            def _forward(x, **kwargs):
                bmt.inspect.record_tensor(x, inspect_name)
                return f(x, **kwargs)
        
        elif modules[k]['type'] == 'post':
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
        super().__init__()
        self.strategies = strategies

    def set_forward(self,
        student,
        teacher,
        forward_fn):
        '''
        Modify the forward function of the student model to compute additional knowledge distillation loss.
        `student` and `teacher` should return (logits, hidden_states, att_scores).
        logits: (batch_size, vocab_size)
        hidden_states: (batch_size, dec_len, hidden_size)
        att_scores: (batch_size, dec_len, enc_len)
        :param student: Student model.
        :param teacher: Teacher model.
        :param foward_fn: Forward function of the student model.
        :param strategies: List of distillation strategies used in the process.
        '''
        # Inspect all necessary tensors
        module_s = {}
        module_t = {}
        strategies = self.strategies
        for strategy in strategies:
            module_s.update(strategy.module_s)
            module_t.update(strategy.module_t)
        inject_inspection(student, module_s, '_student')
        inject_inspection(teacher, module_t, '_teacher')

        def forward(model, dec_input, dec_length, targets, loss_func):
            # Get the output of both student and teacher models.
            with bmt.inspect.inspect_tensor() as inspector:
                outputs = forward_fn(
                    model, dec_input, dec_length, targets, loss_func
                )
                outputs_t = teacher(
                    dec_input, dec_length, return_logits = True
                )
            
            # Get necessary tensors of hidden layers
            records = {}
            for record in inspector._summary:
                records[record['name']] = record['tensor']
            records_s = read_inspection(records, module_s, '_student')
            records_t = read_inspection(records, module_t, '_teacher', detach=True)

            # Calculation of modified loss
            
            if self.training:
                loss = outputs[0]
                d_loss = 0
                logits_s = outputs[1]
                logits_t = outputs_t.detach()
                for strategy in strategies:
                    d_loss = d_loss + strategy.loss(logits_s, logits_t, records_s, records_t)
                loss = loss + d_loss
                outputs[0] = loss
                outputs = outputs + [d_loss, ]

            return outputs

        return forward