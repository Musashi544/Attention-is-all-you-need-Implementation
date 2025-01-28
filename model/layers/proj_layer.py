import torch 
import torch.nn as nn 
import math


# The linear layer after the decoder block 
# Used to map the tokens(embedding) to the vocabulary 
# Projects d_model to vocab_size
class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__() 
        self.proj = nn.Linear(d_model, vocab_size) 
    
    def forward(self, x):
        # (Batch, Seq_len, d_model) --> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1) 