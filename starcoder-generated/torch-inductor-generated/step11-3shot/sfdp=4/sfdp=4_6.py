
from math import sqrt
class SublayerWrapper(torch.nn.Module):
    def __init__(self, size, dropout=0.1):
        super().__init__()
        self.norm = torch.nn.LayerNorm(size)
        self.dropout = torch.nn.Dropout(dropout)
 
    def forward(self, x, sublayer):
        x = sublayer(self.norm(x))
        return self.dropout(x)
 
class PositionwiseFeedforward(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = torch.nn.Linear(d_model, d_ff)
        self.w_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)
 
    def forward(self, x):
        v1 = self.w_1(x)
        v2 = torch.nn.functional.relu(v1)
        v3 = self.dropout(v2)
        v4 = self.w_2(v3)
        return v4

class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, query, value, key, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        v1 = query.view(nbatches, -1, self.heads, self.d_k).transpose(1, 2)
        v2 = key.view(nbatches, -1, self.heads, self.d_k).transpose(1, 2)
        v3 = value.view(nbatches, -1, self.heads, self.d_k).transpose(1, 2)
        
        v4 = self.linears[0](query)
        v5 = self.linears[1](key)
        v6 = self.linears[2](value)
        v7 = self.linears[3](v4)
        v8 = self.linears[4](v5)
        v9 = self.linears[5](v6)
        v10 = self.linears[6](v7)
        v11 = self.linears[7](v8)
        v12 = self.linears[8](v9)
        
        v13 = v10 @ v12.transpose(-2, -1) # Compute the dot product of the query and key, and scale it
        v13 = v13 / sqrt(self.linears[9].weight.size(-1))
        v14 = v13 + v11.unsqueeze(1)
        v15 = torch.nn.functional.softmax(v14, dim=-1) # Apply softmax to the result
        v16 = v15 @ v13 # Compute the dot product of the attention weights and the value
        v17 = v16.transpose(1, 2).contiguous().view(nbatches, -1, self.linears[9].weight.size(-1)) # Return the result
        v18 = self.linears[10](v17)
        return v18

class Model(torch.nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadedAttention(d_model, heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout=dropout)
 
class Encoder(Model):
    def forward(self, x, mask):
        v1 = self.attn(x, x, x, mask=mask)
        v2 = self.feed_forward(v1)
        return v2

x1 = torch.randn(32, 10000, 24)
x2 = torch.randn(32, 1, 24)
m = Encoder(d_model=24, d_ff=48, heads=4)
