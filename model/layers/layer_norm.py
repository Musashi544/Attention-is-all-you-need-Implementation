import torch 
import torch.nn as nn 
import math

class LayerNormalization(nn.Module):
    # eps is epsilon(very small number)
    def __init__(self , features: int, eps : float = 10**-6):
        super().__init__()
        self.eps = eps 
        # output scale Parameter
        self.gamma = nn.Parameter(torch.ones(features)) # Multiplied 
        # Output shift parameter
        self.beta = nn.Parameter(torch.zeroes(features)) # Added 
    
    def forward(self , x):
        # when computing mean , std across a dimension, that dimension is lost. 
        # WHen calculating mean, std for (8 , 7 , 2) matrix across the last dimension, the final result will have shape (8,7)
        # In order to keep that dimension 
        # Use keepdim = True
        mean = x.mean(dim = -1 , keepdim = True) # (batch, seq_len, 1)
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        return self.gamma * ((x - mean) / (std + self.eps)) + self.beta