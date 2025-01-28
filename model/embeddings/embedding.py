import torch 
import torch.nn as nn 
import math
class InputEmbeddings(nn.Module):
    def __init__(self , d_model : int , vocab_size : int): 
        super().__init__()
        # d_model is the dimension of the input vector for each word
        self.d_model = d_model 
        self.vocab_size = vocab_size 
        self.embedding = nn.embedding(vocab_size, d_model) 

    def forward(self , x):
        # mutliplying by sqrt(d_model) specified in section 3.4
        return self.embedding(x) * math.sqrt(self.d_model)