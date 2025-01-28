import torch 
import torch.nn as nn 
import math
from model.layers.layer_norm import LayerNormalization

# Add and Norm
class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout : float): 
        super().__init__() 
        self.dropout = nn.Dropout(dropout) 
        self.norm = LayerNormalization(features)

    # sublayer is the previous layer
    def forward(self, x, sublayer): 
        # Note: this implementation follows 'pre-LN' version of transformer -- which is slightly different from the original transformer in residual connection part. In the original block diagram, the layer normalization(LN) should be applied AFTER multi-head attention / feed-forward network. However, this code applies the LN BEFORE multi-head attention and feed-forward network. You can see the difference by comparing the ResidualConnection forward() code and section 3.2 of original "Attention Is All You Need" paper. This is a valid architecture too (proposed by the other papers), but it is not exactly as proposed in the original one.
        return x + self.dropout(sublayer(self.norm(x)))