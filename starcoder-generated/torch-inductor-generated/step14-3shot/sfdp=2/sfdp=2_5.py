
import torch
import torch.nn as nn
 
class Model(nn.Module):
    def __init__(self, d_model, nhead, dropout_p):
        super().__init__()
        self.attn_module = nn.MultiheadAttention(d_model=d_model, num_heads=nhead, dropout=dropout_p)
 
    def forward(self, x1, x2, x3=None):
        attention_output, attn_mask = self.attn_module(x1, x3, x3, None)
        return attention_output, attn_mask

# Initializing the model
m1 = Model(64, 8, 0.1)

# Inputs to the model
x1 = torch.randn(1, 150, 64)
x2 = torch.randn(1, 150, 64)
x3 = torch.randn(1, 150, 64)
__output__, __attention_mask__ = m1(x1, x2, x3)

