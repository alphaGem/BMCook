from abc import abstractmethod
import types
import torch
import bmtrain as bmt
import torch.nn.functional as F
import model_center

class BMDistillStrategy:
    def __init__(self, scale):
        self.scale = scale
        self.module_s = {}
        self.module_t = {}

    @abstractmethod
    def loss(self, logits_s, logits_t, records_s, records_t):
        pass

class OutputCELoss(BMDistillStrategy):
    def __init__(self, scale, temp):
        '''
        :param scale: Scaler of this strategy.
        :param temp: Temperature of this strategy.
        '''
        super().__init__(scale)
        self.temp = temp

    def loss(self, logits_s, logits_t, records_s, records_t):
        prob_t = F.softmax(logits_t / self.temp, dim=-1)
        log_prob_s = F.log_softmax(logits_s / self.temp, dim=-1)
        return -(prob_t * log_prob_s).sum(dim=1)[:,:-1].mean()*self.scale

class HiddenLoss(BMDistillStrategy):
    def __init__(self, scale,
        projection=False):
        '''
        Calculate the loss by the hidden layers.
        :param scale: Scaler of this strategy.
        :param projection: Whether use projection matrix with a size of (dim_t, dim_s) between models or not.
        :param optimizer: The optimizer used for optimizing the projection matrix.
        :param dim_t: Dimension of the teacher model.
        :param dim_s: Dimension of the student model.
        :param projection_init_method: What init method of projector to use. Choose from 'indentical', 'random' and 'from_embedding'.
        '''
        super().__init__(scale)
        self.projection = projection
        
    def init_w_layer(self,
        optimizer=None,
        dim_t=-1,
        dim_s=-1,
        projection_init_method='identical'):
        if self.projection:
            self.W_layer = model_center.layer.Linear(dim_t, dim_s, init_std=0.02)
            if projection_init_method == 'random':
                pass
            elif projection_init_method == 'identical':
                assert(dim_t == dim_s)
                self.W_layer.weight = bmt.DistributedParameter(torch.eye(dim_s, dtype=torch.half))
        
            bmt.init_parameters(self.W_layer)
            optimizer.add_param_group({'params': self.W_layer.parameters()})
            bmt.synchronize()
        else:
            self.W_layer = lambda x:x

    @abstractmethod
    def loss(self, logits_s, logits_t, records_s, records_t):
        pass

class HiddenMSELoss(HiddenLoss):
    def loss(self, logits_s, logits_t, records_s, records_t):
        loss = 0

        for name_s in self.module_s:
            name_t=self.module_s[name_s]['target']
            h_s = records_s[name_s]
            h_t = records_t[name_t]
            loss += (h_s - self.W_layer(h_t)).pow(2).mean()

        #bmt.print_rank(loss)
        return loss*self.scale
    
    
    def register_layer_pair(self, name_s:str, type_s:str, name_t:str, type_t:str):
        '''
        Match the hidden states between `name_s` of the student model and `name_t` of the teacher model.
        :param name_s: Layer of the student model.
        :type_s: Whether to inspect the student layer before it or after it. Choose from 'pre' and 'post'.
        :param name_t: Layer of the teacher model.
        :type_t: Whether to inspect the teacher layer before it or after it. Choose from 'pre' and 'post'.
        '''
        self.module_s[name_s]={
            "type": type_s,
            "target": name_t
        }
        self.module_t[name_t]={
            "type": type_t
        }

class DynamicHiddenMSELoss(HiddenLoss):
    def loss(self, logits_s, logits_t, records_s, records_t):
        loss = 0

        for name_t in self.module_t:
            best_name = ''
            best_match_loss = torch.tensor(float('inf'), dtype=torch.half)
            for name_s in self.module_s:
                h_s = records_s[name_s].detach()
                h_t = records_t[name_t]
                cur_match_loss = F.mse_loss(h_s, self.W_layer(h_t))
                #bmt.print_rank(name_s[10:20]+':'+name_t[10:20]+'=', cur_match_loss)
                if torch.less(cur_match_loss, best_match_loss):
                    best_match_loss.copy_(cur_match_loss)
                    best_name = name_s
            h_s = records_s[best_name]
            h_t = records_t[name_t]
            match_loss = F.mse_loss(h_s, self.W_layer(h_t))
            loss += match_loss
            # bmt.print_rank(best_name+':'+name_t)

        return loss*self.scale

    def register_layer_student(self, name_s, type_s):
        self.module_s[name_s]={
            "type": type_s
        }
    
    def register_layer_teacher(self, name_t, type_t):
        self.module_t[name_t]={
            "type": type_t
        }