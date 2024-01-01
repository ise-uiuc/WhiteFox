
def gelu(x):
    c = torch.tanh((math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    return x * 0.5 * (1.0 + c)
 
class Model(torch.nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.proj = nn.Linear(size, size)
        self.dropout = dropout
 
    def forward(self, x, mask):
        qkv = self.proj(x).reshape(x.shape[0], -1, 3, 2)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=2), (q, k, v))
        scale_factor = k.shape[-1] ** -0.5
        self_attn = softmax((q @ k.transpose(-2, -1)) * scale_factor)
        self_attn = F.dropout(self_attn, p=self.dropout, training=self.training)
        out = self_attn @ v
        return rearrange(out, 'b h n d -> b n (h d)', h=2)

# Initializing the model
m = Model(size=24, dropout=0.1)

# Inputs to the model
x = torch.randn(1, 24, 100)
attn_mask = torch.randint(0, 1, x.shape, dtype=torch.float)
