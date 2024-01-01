
import torch
import torch.nn as nn
from torch import Tensor
 
class Model(nn.Module):
    def __init__(self, emb_size, heads=8, dropout=0., bias=True):
        super().__init__()
        self.emb_size = emb_size
        self.heads = heads
        self.head_dim = (emb_size // heads)
        self.scale_factor = (self.head_dim ** -0.5)
        self.dropout_p = dropout
        self.qkv = nn.Linear(emb_size, 3 * emb_size, bias=bias)
        dpr = [x.item() for x in torch.linspace(0, dropout, emb_size)]
        self.dropout = nn.Dropout2d(drop=(dpr[0] + dpr[-1]) / 2 if len(dpr) > 1 else dpr[0])
        self.proj = nn.Linear(3 * emb_size, emb_size)
 
    def forward(self, query, key, value, mask=None):
        qkv = self.qkv(query) # Compute the query, key, and value matrices
        qkv_shape = qkv.shape
        (batch_size, seq_length, qkv_length) = qkv_shape[0], qkv_shape[1], qkv_shape[2]
        qkv = qkv.view(batch_size, seq_length, self.heads, 3 * self.head_dim).permute(0, 2, 1, 3) # Reshape and transpose the q, k, and v tensors
        query, key, value = qkv[:, :, :, :self.head_dim], qkv[:, :, :, self.head_dim:2*self.head_dim], qkv[:, :, :, 2*self.head_dim:] # Divide the query, key, and value tensors into query, key, and value blocks
        qk = torch.matmul(query, key.transpose(-2, -1)) # Compute the dot product of the query and the key
        scaled_qk = qk.div(self.scale_factor) # Scale the dot product by the inverse scale factor
        softmax_qk = scaled_qk.softmax(dim=-1) # Apply softmax to the scaled dot product
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=self.dropout_p) # Apply dropout to the softmax output
        output = dropout_qk.matmul(value) # Compute the output using the dot product of the dropout output and the value
        output = output.permute(0, 2, 1, 3) # Transpose the output tensor
        output = output.reshape(batch_size, seq_length, -1) # Reshape the output into a 2-dimensional tensor
        output = self.dropout(output) # Apply a dropout to the reshaped output
        output = self.proj(output) # Apply the projection layer
        return output

# Initializing the model
m = Model(emb_size=256)

# Inputs to the model
x1 = torch.randn(1, 20, 256)
x2 = torch.randn(1, 15, 256)
x3 = torch.randn(1, 15, 256)
