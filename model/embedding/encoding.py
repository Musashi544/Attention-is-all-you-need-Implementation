import torch 
import torch.nn as nn 
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model : int, seq_len : int, dropout : float):
        super().__init__()
        self.d_model = d_model 
        self.seq_len = seq_len # Lenght of the input sentence 
        self.dropout = nn.Dropout(dropout) 

        # Positional encodings wiil be added to this matrix
        pe = torch.zeros(seq_len , d_model)  
        # Create a vector of shape (seq_len)
        # This vector represents the position of each token 
        # unsqueeze adds 1 dimension in the index 1 of the shape 
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        # (d_model / 2)
        dim = torch.arange(0, d_model, 2).float() # 2i

        # Apply sin at even position 
        pe[: , 0::2] = torch.sin(position / 10_000.0 ** (dim / d_model)) # sin(position * (10000 ** (2i / d_model))
        # Apply cos at odd position
        pe[: , 1::2] = torch.cos(position / 10_000.0 ** (dim / d_model)) # cos(position * (10000 ** (2i / d_model))

        # Add a batch dimension to the positional encoding
        pe.unsqueeze(0) # (1, seq_len , d_model)

        # Register the positional encoding as a buffer
        # These are parameters that should be saved and restored but are not trained by the optimizer. 
        # Buffer wont be returned by model.parameters
        self.register_buffer('pe', pe) 
    
    def forward(self , x):
        # Adds the positional encoding of each token with its respective embedding
        # (batch_size , seq_len , d_model) 
        x = x + (self.pe[: , :x.shape[1] , :]).requires_grad(False) 

        return self.dropout(x)