
class SelfAttention(torch.nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, dropout_p=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.proj = torch.nn.Linear(dim, dim)
 
    def forward(self, x):
        b = x.shape[0]
        qkv = self.qkv(x).reshape(b, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, -1, self.dim)
        x = self.proj(x)
        return x

# Initializing the model
m = SelfAttention(dim=5)

# Inputs to the model
x1 = torch.randn(1, 4, 5)
