import torch 
import torch.nn as nn 
import math

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model 
        self.h = h # no of heads 

        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # dimension of vector seen by each head 

        self.w_q = nn.Linear(d_model , d_model , bias = False)
        self.w_k = nn.Linear(d_model , d_model , bias = False)
        self.w_v = nn.Linear(d_model , d_model , bias = False)

        self.w_o = nn.Linear(d_model , d_model , bias = False)

        self.dropout = nn.Dropout(dropout) 

    @staticmethod 
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) --> (batch, h, seq_len, seq_len)
        attention_score = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)

        if mask is not None: 
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_score.masked_fill(mask == 0, -1e9)

        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1) # Apply softmax

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) --> (batch, h, seq_len, d_k)
        # Returning  tuple to pass to other blocks 
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores
    

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch,seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k) # (batch,seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v) # (batch,seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)

        # We are transposing because we want to get (seq_len , d_k) for every head to perform the self_attention
        # with shape (batch, h, seq_len, d_k) means that each head can only see a portion of the embeddings which is d_k
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        # calculate attention
        x, attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)  
        return self.w_o(x)