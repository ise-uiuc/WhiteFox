
class Model(torch.nn.Module):
    def __init__(self, dim_hidden, num_heads, dropout_p=0.1):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.num_heads = num_heads
        self.dropout_p = dropout_p
 
        self.qkv_proj = torch.nn.Linear(dim_hidden, dim_hidden * 3)
        self.attn_dropout = torch.nn.Dropout(dropout_p)        
        self.out_proj = torch.nn.Linear(dim_hidden, dim_hidden)
 
    def forward(self, x, x, x):
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)
 
        q, k, v = q.reshape(-1, q.size(-1)).transpose(-2, -1), k.reshape(-1, k.size(-1)).transpose(-2, -1), v.reshape(-1, v.size(-1)).transpose(-2, -1)
 
        inv_sqrt_k = 1 / math.sqrt(k.size(-1))
        attn = torch.softmax((q @ k.transpose(-2, -1)) * inv_sqrt_k, dim=-1)
        attn = self.attn_dropout(attn)
 
        out = attn.matmul(v)
 
        out = out.reshape(x.size(0), x.size(1), x.size(2))
        out = self.out_proj(out)
        return out, attn

# Initializing the model
m = Model(512, 8)

# Inputs to the model
x = torch.randn(32, 128, 512)
y = torch.randn(32, 128, 512)
z = torch.randn(32, 128, 512)
