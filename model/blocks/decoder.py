import torch 
import torch.nn as nn 
import math
from model.layers.multi_head_attention import MultiHeadAttentionBlock 
from model.layers.feed_forward_network import FeedForwardBlock 
from model.layers.residual_connection import ResidualConnection
from model.layers.layer_norm import LayerNormalization

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block 
        self.cross_attention_block = cross_attention_block 
        self.feed_forward_block = feed_forward_block 
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask)) 
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask)) 
        x = self.residual_connections[2](x, self.feed_forward_block) 
        return x 

class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList): 
        super().__init__() 
        self.layers = layers 
        self.norm = LayerNormalization(features) 
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)