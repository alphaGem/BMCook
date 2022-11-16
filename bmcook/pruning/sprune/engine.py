import torch
import bmtrain as bmt
from typing import Dict
from torch.nn.parameter import Parameter
from .func import determinate_mask, sample, binarize
from .plugin import SPrunePlugin


class SPruneStrategy:
    def __init__(self, config: Dict) -> None:
        # self.criterion = config['criterion']
        # assert self.criterion == 'l0', "BMCook sprune do not support other criterions besides l0 yet."
        self.fixed_mask_path = config['fixed_mask_path']
        self.training_mask = config['training_mask']
        self.mask_mode = config['mask_mode']
        # self.target_mode = config['target_mode']
        # self.target_sparsity = config['target_sparsity']


class SPruneEngine:
    r"""
    SPruneEngine is used for the mask computation and update of SPrunePlugin.

    The engine design is based on L0 regularization method and a lagrangian term. For L0 regularization details, see paper
        "Learning Sparse Neural Networks through L_0 Regularization" <https://openreview.net/forum?id=H1Y8hhg0b>.
        For lagrangian term in PLM structure pruning, see paper "Structured Pruning of Large Language Models" 
        <https://arxiv.org/abs/1910.04732>.
    """
    def __init__(self, config: Dict, plugin: SPrunePlugin) -> None:
        r"""Init the SpruneEngine from a SPrunePlugin. It will initilize all the :class:`torch.nn.Parameter`
        used for learning the sprune mask, and create the optimizer for l0 regularization.

        Args:
            config: `(Dict)`, the sprune config.
            plugin: `(SPrunePlugin)`, the SPrunePlugin.
        """
        super().__init__()
        self.strategy = SPruneStrategy(config)
        self.target_sparsity = self.strategy.target_sparsity
        self.plugin = plugin
        self.training = True

        self.training_loga = {}
        for mask in self.strategy.training_mask:
            shape = self.plugin.info_to_engine['shape'][mask]
            self.training_loga[mask+'_loga'] = Parameter(torch.empty(shape[0], dtype=torch.float, device='cuda').normal_(0., 1e-2))

        self.create_sprune_optimizer()

    def create_sprune_optimizer(self):
        r"""Create the sprune optimizer and lagrangian optimizer, making the learning of loga and 
        lagrangian terms to be an adversarial game.
        
        sprune optimizer will manage the loga parameters.

        lagrangian optimizer will manage the lagrangian terms.
        """
        l0_params = [{
                        "params": [p for _, p in self.training_loga.items()],
                        "weight_decay": 0.0,
                        "lr": 0.1
                        }]
        self.sp_optimizer = torch.optim.AdamW(l0_params)
    
    def update(self):
        r"""
        update the sprune parameters and lagrangian parameters.
        """
        if self.training:
            info_list = self.update_plugin_mask(training=True)
            loss, sparsity = self.loss(info_list)
            if torch.abs(sparsity - self.target_sparsity) < 5e-5:
                bmt.print_rank("binarize the mask and begin finetune...")
                info_list = self.update_plugin_mask(training=False)
                for v in self.training_loga.values():
                    v.requires_grad_(False)
                self.training = False
        else:
            info_list = self.update_plugin_mask(training=False)
            loss, sparsity = self.loss(info_list)
        return loss, sparsity

    def update_plugin_mask(self, training: bool = True):
        r"""update the mask managed in plugin"""
        info_list = {}
        for k, v in self.training_loga.items():
            module = k.split('_loga')[0]

            mask = sample(v) if training is True else binarize(v)
            train_mask = determinate_mask(v)
            assert mask.size(0) == train_mask.size(0)
            
            for index in range(mask.size(0)):
                self.plugin.__dict__[module][index]['mask'] = mask[index].clone().detach()
                
                param = self.plugin.__dict__[module][index]['param']
                index_all = self.plugin.__dict__[module][index]['index']

                if index_all not in info_list:
                    info_list[index_all] = {'module': [module], 'param': [param], 'score': [train_mask[index]]}
                else:
                    if module in info_list[index_all]['module']:
                        module_correct = 'cross_' + module
                    else:
                        module_correct = module
                    info_list[index_all]['module'].append(module_correct)
                    info_list[index_all]['param'].append(param)
                    info_list[index_all]['score'].append(train_mask[index])

        return info_list

    '''
    def lagrangian_loss_dimension(self):
        r"""calculate the lagrangian loss to get the target dimension"""
        dimension_score = determinate_mask(self.training_loga)
        all_dimension = dimension_score.size(1)
        
        expected_dimension = torch.sum(dimension_score, -1)
        loss_dimension = torch.sum((self.target_dimension - expected_dimension) / all_dimension)
        
        lagrangian_loss = self.lambda_1 * loss_dimension + self.lambda_2 * (loss_dimension ** 2)
        
        return lagrangian_loss, expected_dimension
    '''