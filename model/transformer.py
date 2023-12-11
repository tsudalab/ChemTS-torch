import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerBlock(nn.Module):
    def __init__(self, conf):
        super().__init__()
        
        embed_dim = conf['Model']['embed_dim']
        dropout_rate = conf['Model']['dropout_rate']
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(conf)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout_rate))
        
    def forward(self, x):
        x = x + self.attention(x)   # residual
        x = x + self.mlp(x)
        return x
        

class SelfAttention(nn.Module):
    def __init__(self, conf):
        super().__init__()
        
        embed_dim = conf['Model']['embed_dim']
        n_heads = conf['Model']['n_heads']
        seq_len = conf['Data']['seq_len']
        if 'sub_len' in conf['Data'].keys():
            seq_len += conf['Data']['sub_len']
        dropout_rate = conf['Model']['dropout_rate']
        assert embed_dim % n_heads == 0
        
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.n_heads = n_heads
        self.register_buffer("mask", 
            torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len))
        
    
    def forward(self, x):
        B, L, D = x.shape
        
        k = self.key(x).view(B, L, self.n_heads, -1).transpose(1, 2)    # batch x n_heads x len x lat_dim
        q = self.query(x).view(B, L, self.n_heads, -1).transpose(1, 2)  # batch x n_heads x len x lat_dim
        v = self.value(x).view(B, L, self.n_heads, -1).transpose(1, 2)  # batch x n_heads x len x lat_dim
        
        s = (q @ k.transpose(-1, -2)) / (k.size(-1) ** 0.5)     # batch x n_heads x len x len
        s = s.masked_fill(self.mask[:,:,:L,:L] == 0, float('-inf'))    # apply mask
        s = self.dropout(F.softmax(s, dim=-1))
        
        y = s @ v   # batch x n_heads x len x lat_dim
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        
        y = self.dropout(self.out_proj(y))
        
        return y
        
    

class Transformer(nn.Module):
    def __init__(self, conf):
        super().__init__()
        
        self.conf = conf
        
        self.vocab_len = conf['Data']['vocab_len']
        self.seq_len = conf['Data']['seq_len']
        self.embed_dim = conf['Model']['embed_dim']
        self.dropout_rate = conf['Model']['dropout_rate']
        self.n_layers = conf['Model']['n_layer']
        self.hidden_dim = conf['Model']['hidden_dim']
        self.n_heads = conf['Model']['n_heads']
        
        # define layers
        self.token_embedding = nn.Embedding(self.vocab_len, self.embed_dim)
        self.pos_embedding = nn.Parameter(
            torch.zeros((1, self.seq_len, self.embed_dim))) # learnable position embedding
        self.dropout = nn.Dropout(self.dropout_rate)
        
        self.blocks = nn.ModuleList(
            [TransformerBlock(conf) for _ in range(self.n_layers)])
        
        self.norm = nn.LayerNorm(self.embed_dim)
        self.output = nn.Linear(self.embed_dim, self.vocab_len)
        
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')
        
        self.apply(self._init_weights)
        
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.xavier_uniform(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, inputs, target=None, length=None):

        B, L = inputs.size()
        token_embeddings = self.token_embedding(inputs)
        pos_embeddings = self.pos_embedding[:,:L,:]
        x = self.dropout(token_embeddings + pos_embeddings)
        
        for block in self.blocks:
            x = block(x)
        
        logits = self.output(self.norm(x))
        
        loss = None
        if target is not None:
            loss = self.cross_entropy_loss(logits.view(-1, logits.size(-1)), target.view(-1))
        
        return logits, target, loss
        
    @torch.no_grad()
    def sample(self, batch_size, bos, eos, device, max_length=73):
        """Use this function if device is GPU"""

        x = torch.ones(
            (batch_size, 1), 
            dtype=torch.long, 
            device=device) * bos
        
        for _ in range(max_length):
            logits, _, _ = self(x)
            logits = logits[:, -1, :]
            
            probs = torch.softmax(logits, dim=-1)
            x_new = torch.multinomial(probs, num_samples=1)
            
            x = torch.cat([x, x_new], dim=1)
        return x.cpu().numpy()
    
    @torch.no_grad()
    def predict(self, ints, device):
        x = torch.tensor(ints, device=device).view(1, -1)

        logits, _, _ = self(x)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

        return probs.squeeze().cpu().numpy()
    
class PositionalEncoding(nn.Module):
    """Add positional encoding to embedding

    Codes modified from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        # x: batch, seq_len, embedding_dim
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
    
class ConditionTransformer(Transformer):
    def __init__(self, conf):
        super().__init__(conf)
        
        # type embedding: substructure: 0 / molecule : 1
        self.sub_pos_embedding = PositionalEncoding(self.embed_dim)
        self.type_embedding = nn.Embedding(2, self.embed_dim)
        
    def forward(self, inputs, target=None, length=None):
        
        sub, mols = inputs

        B, L = mols.size()
        token_embeddings = self.token_embedding(mols)
        pos_embeddings = self.pos_embedding[:,:L,:]
        type_embeddings = self.type_embedding(torch.ones_like(mols))
        
        x = self.dropout(token_embeddings + pos_embeddings + type_embeddings)
        
        # add condition information
        _, S = sub.size()
        sub_tok_embeddings = self.token_embedding(sub)
        sub_pos_embeddings = self.sub_pos_embedding(sub_tok_embeddings)
        sub_type_embeddings = self.type_embedding(torch.zeros_like(sub))
        
        sub_x = self.dropout(sub_pos_embeddings + sub_type_embeddings)
        
        x = torch.cat([sub_x, x], dim=1)
        
        
        for block in self.blocks:
            x = block(x)
        
        logits = self.output(self.norm(x))
        logits = logits[:,S:,:]
        
        loss = None
        if target is not None:
            loss = self.cross_entropy_loss(logits.contiguous().view(-1, logits.size(-1)), target.view(-1))
        
        return logits, target, loss
    
    @torch.no_grad()
    def sample_from_substruct(self, batch_size, bos, eos, device, cond_ints, max_length=73):
        """Use this function if device is GPU"""

        x = torch.ones(
            (batch_size, 1), 
            dtype=torch.long, 
            device=device) * bos
        x_cond = torch.tensor(cond_ints).view(1, -1).expand(batch_size, -1).to(x)


        for _ in range(max_length):
            x_inputs = (x_cond, x)

            logits, _, _ = self(x_inputs)
            logits = logits[:, -1, :]
            
            probs = torch.softmax(logits, dim=-1)
            x_new = torch.multinomial(probs, num_samples=1)
            
            x = torch.cat([x, x_new], dim=1)
        return x.cpu().numpy()