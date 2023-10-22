from typing import Any
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Optimizer
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy

class RNNModel(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.embedding = nn.Embedding(
            num_embeddings=conf['Data']['vocab_len'],
            embedding_dim=conf['Data']['vocab_len'])
        self.gru = nn.GRU(
            input_size=conf['Data']['vocab_len'],
            hidden_size=conf['Model']['units'],
            num_layers=2,
            batch_first=True,
            dropout=conf['Model']['dropout_rate'])
        self.linear = nn.Linear(conf['Model']['units'], conf['Data']['vocab_len'])

    def forward(self, x, lengths):
        print(x)
        print(x.shape)
        x = self.embedding(x)
        x = pack_padded_sequence(
            input=x, 
            lengths=lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        x, _ = self.gru(x) # x: batch x seq_len x latent, hidden: n_layers x batch x latent

        # x_shape = x.shape
        # x = self.linear(x.contiguous().view(-1, x_shape[-1]))
        # x =  x.contiguous().view(x_shape[0], x_shape[1], -1)
        x = self.linear(x.data)
        return x
    
    @torch.no_grad()
    def sample(self, batch_size, bos, eos, device, max_length=140):
        """Use this function if device is GPU"""

        bos = torch.ones(
            (batch_size, 1), 
            dtype=torch.long, 
            device=device) * bos

        # sample first output
        output = []
        x = self.embedding(bos)
        x, hidden = self.gru(x)
        x = self.linear(x)
        x = torch.softmax(x, dim=-1)
        x = torch.multinomial(x.squeeze(), 1)
        output.append(x)

        # a tensor to indicate if the <eos> token is found
        # for all data in the mini-batch
        finish = torch.zeros(batch_size, dtype=torch.bool).to(device)

        # sample until every sequence in the mini-batch
        # has <eos> token
        for _ in range(max_length):
            # forward rnn
            x = self.embedding(x)
            x, hidden = self.gru(x, hidden)
            x = self.linear(x)
            x = torch.softmax(x, dim=-1)
            
            # sample
            x = torch.multinomial(x.squeeze(), 1)
            output.append(x)

            # terminate if <eos> is found for every data
            eos_sampled = (x == eos).data
            finish = torch.logical_or(finish, eos_sampled.squeeze())
            if torch.all(finish):
                return torch.cat(output, -1)

        return torch.cat(output, -1).cpu().detach().numpy()
    
class SmilesPredModule(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()

        self.conf = conf
        
        self.rnn_model = RNNModel(conf)
        self._init_weights(self.rnn_model)
        print(self.rnn_model)
        
        self.lr = conf['Train']['learning_rate']
        self.optimizer = conf['Train']['optimizer']
        self.scheduler = conf['Train']['scheduler']
        self.decay_steps = conf['Train']['decay_steps']
        self.decay_alpha = conf['Train']['decay_alpha']
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')
        self.accuracy = MulticlassAccuracy(conf['Data']['vocab_len'])
        
    def _init_weights(self, model):
        for param in model.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)
        
    def training_step(self, batch, batch_idx):
        inputs, target, lengths = batch
        output = self.rnn_model(inputs, lengths) # batch x seq_len x vocab_len
        # loss = self.cal_loss(output, target) # batch x seq_len x vocab_len
        
        target = pack_padded_sequence(
            target,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False).data

        loss = self.cross_entropy_loss(output, target)
        acc = self.cal_acc(output, target)
        
        self.log_dict({"train_loss": loss, "train_acc": acc},
            prog_bar=True, logger=True, on_step=False, on_epoch=True, 
            batch_size=self.conf['Train']['batch_size'])
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, target, lengths = batch
        output = self.rnn_model(inputs, lengths) # batch x seq_len x vocab_len
        # loss = self.cal_loss(output, target) # batch x seq_len x vocab_len
        
        target = pack_padded_sequence(
            target,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False).data

        loss = self.cross_entropy_loss(output, target)
        acc = self.cal_acc(output, target)
        
        self.log_dict({"val_loss": loss, "val_acc": acc},
            prog_bar=True, logger=True, on_step=False, on_epoch=True,
            batch_size=self.conf['Train']['batch_size'])
        return loss
    
    @torch.no_grad()
    def predict(self, x):
        self.rnn_model.eval()
        # return nn.functional.softmax(self.rnn_model(x), dim=2
        #     )[:,-1:,:].detach().cpu().numpy()
        return self.rnn_model.pred_prob(x).detach().cpu().numpy()
    
    def cal_loss(self, output, target):
        output = output.view(-1, output.size(-1))
        target = target.view(-1)
        loss = self.cross_entropy_loss(output, target)
        return loss
    
    def cal_acc(self, output, target):
        target = target.view(-1)
        
        probs = nn.functional.softmax(output.view(-1, output.size(-1)), dim=1)
        preds = probs.argmax(dim=1)
        acc = torch.sum(preds == target) / target.shape[0]

        return acc
        
    def configure_optimizers(self):
        """Configure the optimizer
        """
        if self.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.rnn_model.parameters(),
                lr=self.lr,
            )
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.rnn_model.parameters(),
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
                verbose=True
            )
        else:
            raise RuntimeError(
                f'Currenly scheduler only supports CosineAnnealingLR, but {self.scheduler} is set.')
        return [optimizer,], [scheduler,]