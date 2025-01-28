import torch 
import torch.nn as nn 
import math

class FeedForwardBlock(nn.Module):
    # THe network contains input and output layer and 1 hidden layer 
    # input_dim = output_dim = d_model = 512
    # hidden_dim = d_ff = 2048
    def __init__(self , d_model : int, d_ff : int, dropout : float):
        super().__init__() 
        self.linear_1 = nn.Linear(d_model , d_ff) # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff , d_model) # W2 and b2
    
    def forward(self , x):
        # (batch_size, seq_len, d_model) * (d_model , d__ff) -> (batch_size , seq_len, d_ff) * (d_ff , d_model) -> (batch_size, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))