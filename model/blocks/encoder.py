import torch 
import torch.nn as nn 
import math
from model.layers.multi_head_attention import MultiHeadAttentionBlock 
from model.layers.feed_forward_network import FeedForwardBlock 
from model.layers.residual_connection import ResidualConnection
from model.layers.layer_norm import LayerNormalization

class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block 
        self.feed_forward_block = feed_forward_block 
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
    
    # src_mask is applied to the input so that there is not interaction with the padding tokens 
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x : self.self_attention_block(x, x, x, src_mask)) 
        x = self.residual_connections[1](x, self.feed_forward_block) 
        return x 

# This class represents n times encoder block where the output of encoder block is sent to the other encoder block.
class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers # These are the n encoder bocks 
        self.norm = LayerNormalization(features) 

    def forward(self, x, mask):
        for layer in self.layers: 
            x = layer(x, mask) 
        return self.norm(x)