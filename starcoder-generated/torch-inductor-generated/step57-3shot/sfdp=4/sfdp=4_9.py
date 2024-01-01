
import torch
import torch.nn as nn



class Model(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output
# Inputs to the model
query = torch.randn(1, 64, 56, 56)
key = torch.randn(1, 64, 56, 56)
value = torch.randn(1, 64, 56, 56)
attn_mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)

query1 = torch.randn(1,64,56,56)
key2 = torch.randn(1, 64, 56, 56)
valuee = torch.randn(1, 64, 56, 56)
maskk = (torch.torch.rand(1, 56, 56) > 0.7).fill_(-1000000000,0.0)
