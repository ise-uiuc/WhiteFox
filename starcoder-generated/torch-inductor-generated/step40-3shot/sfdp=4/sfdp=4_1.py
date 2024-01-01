
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v, mask):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k2, v2, mask):
        qk = q @ k2.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v2
        return output
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x5, q5, k, v, mask):
        qk = x5 @ k.transpose(-2, -1) / math.sqrt(x5.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x2, k, v, mask):
        qk = x2 @ k.transpose(-2, -1) / math.sqrt(x2.size(-1))
        qk = qk + mask
        attn_mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x5, n5, k, v, mask):
        qk = x5 @ k.transpose(-2, -1) / math.sqrt(x5.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, k2, v2, q, mask):
        qk = q @ k2.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v2
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
