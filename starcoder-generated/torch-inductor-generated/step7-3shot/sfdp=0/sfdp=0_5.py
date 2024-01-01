
class Model(torch.nn.Module):
    def __init__(self, d1, d2, h, n, causal_attention):
        super().__init__()

    def forward(self, x1, x2, x3):
        q = x2.float()
        k = x3.float()
        v = x1.float()
        inv_scale = 1. / math.sqrt(k.size(-1))
        attn = torch.matmul(q, k.transpose(-2, -1)) * inv_scale
        attn = torch.softmax(attn, dim=-1)
        x = torch.matmul(attn, v)
        return x

# Initializing the model
m = Model(32, 32, 64, 4, True)

# Inputs to the model
x1 = torch.randn(1, 4, 32, 32)
x2 = torch.randn(1, 4, 32, 32)
x3 = torch.randn(1, 4, 32, 32)
