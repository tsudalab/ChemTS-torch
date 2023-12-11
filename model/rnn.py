from typing import Any
import math

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim import Optimizer

class RNNModel(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.embedding = nn.Embedding(
            num_embeddings=conf['Data']['vocab_len'],
            embedding_dim=conf['Data']['vocab_len'])
        self.gru = nn.GRU(
            input_size=conf['Data']['vocab_len'],
            hidden_size=conf['Model']['hidden_dim'],
            num_layers=conf['Model']['n_layer'],
            batch_first=True,
            dropout=conf['Model']['dropout_rate'])
        self.linear = nn.Linear(conf['Model']['hidden_dim'], conf['Data']['vocab_len'])
        
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')


    def forward(self, inputs, target, lengths):

        x = self.embedding(inputs)
        x = pack_padded_sequence(
            input=x, 
            lengths=lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        x, _ = self.gru(x) # x: batch x seq_len x latent, hidden: n_layers x batch x latent
        x = self.linear(x.data)
        
        target = pack_padded_sequence(
            target,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False).data
        
        loss = self.cross_entropy_loss(x, target)
        
        return x, target, loss
    
    @torch.no_grad()
    def predict(self, ints, device):
        ints = torch.tensor(ints, device=device)
        token_id = torch.ones((1, 1), 
            dtype=torch.long, 
            device=device) * ints[0]
        
        x = self.embedding(token_id)
        x, hidden = self.gru(x)

        for i in range(1, ints.size(0)):
            x = self.embedding(ints[i].view(1, -1))
            x, hidden = self.gru(x, hidden)
        x = torch.softmax(self.linear(x), dim=-1)
        return x.squeeze().cpu().numpy()
    
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