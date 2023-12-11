import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

"""
Implementation of the transformer decoder.
Codes modified from https://wingedsheep.com/building-a-language-model/.
"""

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
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
    
class MaskedSelfAttention(torch.nn.Module):
    """
    Pytorch module for a self attention layer.
    This layer is used in the MultiHeadedSelfAttention module.

    Input dimension is: (batch_size, sequence_length, embed_dim)
    Output dimension is: (batch_size, sequence_length, head_dim)
    """

    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.query_layer = nn.Linear(embed_dim, self.head_dim)
        self.key_layer = nn.Linear(embed_dim, self.head_dim)
        self.value_layer = nn.Linear(embed_dim, self.head_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        """
        Compute the self attention.

        x dimension is: (batch_size, sequence_length, embed_dim)
        output dimension is: (batch_size, sequence_length, head_dim)
        mask dimension is: (batch_size, sequence_length)

        mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
        """

        # x dimensions are: (batch_size, sequence_length, embed_dim)
        # query, key, value dimensions are: (batch_size, sequence_length, head_dim)
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        # Calculate the attention weights.
        # attention_weights dimensions are: (batch_size, sequence_length, sequence_length)
        attention_weights = torch.matmul(query, key.transpose(-2, -1))

        # Scale the attention weights.
        attention_weights = attention_weights / (self.head_dim ** 0.5)

        # Apply the mask to the attention weights, by setting the masked tokens to a very low value.
        # This will make the softmax output 0 for these values.
        mask = mask.reshape(attention_weights.shape[0], 1, attention_weights.shape[2])
        attention_weights = attention_weights.masked_fill(mask == 0, -1e9)

        # Softmax makes sure all scores are between 0 and 1 and the sum of scores is 1.
        # attention_scores dimensions are: (batch_size, sequence_length, sequence_length)
        attention_scores = self.softmax(attention_weights)

        # The attention scores are multiplied by the value
        # Values of tokens with high attention score get highlighted because they are multiplied by a larger number,
        # and tokens with low attention score get drowned out because they are multiplied by a smaller number.
        # Output dimensions are: (batch_size, sequence_length, head_dim)
        return torch.bmm(attention_scores, value)
    
    
    
class MaskedMultiHeadedSelfAttention(torch.nn.Module):
    """
    Pytorch module for a multi head attention layer.

    Input dimension is: (batch_size, sequence_length, embed_dim)
    Output dimension is: (batch_size, sequence_length, embed_dim)
    """

    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dimension = embed_dim // n_heads
        self.n_heads = n_heads

        # Create the self attention modules
        self.self_attentions = torch.nn.ModuleList(
            [MaskedSelfAttention(embed_dim, self.head_dimension) for _ in range(n_heads)])

        # Create a linear layer to combine the outputs of the self attention modules
        self.output_layer = torch.nn.Linear(n_heads * self.head_dimension, embed_dim)

    def forward(self, x, mask):
        """
        Compute the multi head attention.

        x dimensions are: (batch_size, sequence_length, embed_dim)
        mask dimensions are: (batch_size, sequence_length)
        mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
        """
        # Compute the self attention for each head
        # self_attention_outputs dimensions are:
        # (n_heads, batch_size, sequence_length, head_dimension)
        self_attention_outputs = [self_attention(x, mask) for self_attention in self.self_attentions]

        # Concatenate the self attention outputs
        # self_attention_outputs_concatenated dimensions are:
        # (batch_size, sequence_length, n_heads * head_dimension)
        concatenated_self_attention_outputs = torch.cat(self_attention_outputs, dim=2)

        # Apply the output layer to the concatenated self attention outputs
        # output dimensions are: (batch_size, sequence_length, embed_dim)
        return self.output_layer(concatenated_self_attention_outputs)

class DecoderLayer(torch.nn.Module):
    """
    Pytorch module for an encoder layer.

    An encoder layer consists of a multi-headed self attention layer, a feed forward layer and dropout.

    Input dimension is: (batch_size, sequence_length, embedding_dimension)
    Output dimension is: (batch_size, sequence_length, embedding_dimension)
    """

    def __init__(
            self,
            embed_dim,
            n_heads,
            forward_dim,
            dropout_rate
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.forward_dim = forward_dim
        self.dropout_rate = dropout_rate

        self.multi_headed_self_attention = MaskedMultiHeadedSelfAttention(embed_dim, n_heads)
        self.feed_forward = FeedForward(embed_dim, forward_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        """
        Compute the encoder layer.

        x dimensions are: (batch_size, sequence_length, embedding_dimension)
        mask dimensions are: (batch_size, sequence_length)
        mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
        """

        # Layer normalization 1
        normalized_x = self.norm1(x)

        # Multi headed self attention
        attention_output = self.multi_headed_self_attention(normalized_x, mask)

        # Residual output
        residual_output = x + attention_output

        # Layer normalization 2
        normalized_residual_output = self.norm2(residual_output)

        # Feed forward
        feed_forward_output = self.feed_forward(normalized_residual_output)

        feed_forward_output = self.dropout(feed_forward_output)

        # Residual output
        return residual_output + feed_forward_output

class DecoderStack(torch.nn.Module):
    """
    A stack of decoders.
    """

    def __init__(
            self,
            embed_dim,
            n_layers,
            n_heads,
            forward_dim,
            dropout_rate,
            seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.forward_dim = forward_dim
        self.dropout_rate = dropout_rate
        self.seq_len = seq_len

        # Create the encoder layers
        self.encoder_layers = torch.nn.ModuleList(
            [DecoderLayer(embed_dim, n_heads, forward_dim, dropout_rate) for _ in
             range(n_layers)])

    def forward(self, x, mask):
        decoder_outputs = x
        for decoder_layer in self.encoder_layers:
            decoder_outputs = decoder_layer(decoder_outputs, mask)

        return decoder_outputs


class FeedForward(torch.nn.Module):
    """
    A feed forward layer.
    """

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        return self.linear_2(torch.relu(self.linear_1(x)))


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
        self.n_heads = conf['Model']['n_head']
        
        
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_len,
            embedding_dim=self.embed_dim)
        
        self.position_encoding = PositionalEncoding(
            d_model=self.embed_dim,
            dropout=self.dropout_rate,
            max_len=self.seq_len)
        
        self.norm = nn.LayerNorm(self.embed_dim)

        self.decoder = DecoderStack(
            embed_dim=self.embed_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            forward_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            seq_len=self.seq_len)
        
        self.linear = nn.Linear(self.embed_dim, self.vocab_len)
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
    def forward(self, inputs, target, length=None):
        # calculate mask
        mask = torch.ones_like(inputs)
        mask[inputs == 0] = 0

        x = self.embedding(inputs)
        x = self.position_encoding(x)
        x = self.norm(x)
        
        x = self.decoder(x, mask)
        x = self.linear(x)
        
        loss = self.cross_entropy_loss(x.view(-1, x.size(-1)), target.view(-1))

        return x, loss
    
    @torch.no_grad()
    def pred_next(self, x, mask):
        logits, _ = self(x, mask)
        prob = torch.softmax(logits, dim=-1)
        return prob[:, -1]
    
    @torch.no_grad()
    def sample(self, batch_size, bos, eos, device, max_length=73):
        start = [0 for _ in range(max_length - 1)] + [bos]
        
        # start = torch.ones(
        #     (batch_size, 1), 
        #     dtype=torch.long, 
        #     device=device) * bos
        x = torch.tensor(start,
            dtype=torch.long, 
            device=device).view(1, -1)
        out = x
        for _ in range(max_length):

            x = out[:, -max_length:]

            mask = torch.ones_like(x)
            mask[x == 0] = 0

            # Compute the next token probabilities
            next_token_probabilities = self.pred_next(
                x=x,
                mask=mask)

            # Sample the next token from the probability distribution
            next_token = torch.multinomial(next_token_probabilities, num_samples=1)

            # Append the next token to the output
            out = torch.cat([out, next_token], dim=1)

            # If the end of sequence token is reached, stop generating tokens
            if next_token == eos:
                break

        return out.cpu().detach().numpy()