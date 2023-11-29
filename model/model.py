from typing import Any
import math

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Optimizer
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy

from model.rnn import RNNModel
from model.transformer import Transformer, ConditionTransformer
    

class StringPredModule(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()

        self.conf = conf
        
        if conf['Model']['type'].lower() == 'rnn':
            self.model = RNNModel(conf)
        elif conf['Model']['type'].lower() == 'transformer':
            self.model = Transformer(conf)
        elif conf['Model']['type'].lower() == 'transformer_cond':
            self.model = ConditionTransformer(conf)
        else:
            raise ValueError(f"Model type should be 'rnn' or 'transformer',but {conf['Model']['type']} is set.")
        self._init_weights(self.model)
        print(self.model)
        
        self.lr = conf['Train']['learning_rate']
        self.optimizer = conf['Train']['optimizer']
        self.scheduler = conf['Train']['scheduler']
        self.decay_steps = conf['Train']['decay_steps']
        self.decay_alpha = conf['Train']['decay_alpha']
        
    def _init_weights(self, model):
        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)
        
    def training_step(self, batch, batch_idx):
        output, target, loss = self.model(*batch) # batch x seq_len x vocab_len, batch: (inputs, target, lengths)     
        acc = self.cal_acc(output, target)
        
        self.log_dict({"train_loss": loss, "train_acc": acc},
            prog_bar=True, logger=True, on_step=True, on_epoch=True, 
            batch_size=self.conf['Train']['batch_size'])
        return loss
    
    def validation_step(self, batch, batch_idx):
        output, target, loss = self.model(*batch) # batch x seq_len x vocab_len, batch: (inputs, target, lengths)     
        acc = self.cal_acc(output, target)
        
        self.log_dict({"val_loss": loss, "val_acc": acc},
            prog_bar=True, logger=True, on_step=False, on_epoch=True,
            batch_size=self.conf['Train']['batch_size'])
        return loss
    
    # @torch.no_grad()
    # def predict(self, x):
    #     self.model.eval()

    #     return self.model.pred_prob(x).detach().cpu().numpy()
    
    def cal_acc(self, output, target):
        target = target.view(-1)
        
        probs = torch.softmax(output.contiguous().view(-1, output.size(-1)), dim=1)
        preds = probs.argmax(dim=1)
        acc = torch.sum(preds == target) / target.shape[0]

        return acc
        
    def configure_optimizers(self):
        """Configure the optimizer
        """
        if self.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
            )
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
            )
        
        else:
            raise RuntimeError(
                f'Currenly optimizer only supports Adam and AdamW, but {self.optimizer} is set.')
        if self.scheduler.lower() == 'cosineannealinglr':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.decay_steps,
                eta_min=self.lr * self.decay_alpha,
                verbose=False
            )
        else:
            raise RuntimeError(
                f'Currenly scheduler only supports CosineAnnealingLR, but {self.scheduler} is set.')
        return [optimizer,], [scheduler,]