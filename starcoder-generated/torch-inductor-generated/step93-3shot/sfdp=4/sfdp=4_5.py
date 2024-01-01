
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q0, k2, v1, mask_val):
        qk = q0 @ torch.transpose(k2, -2, -1)
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v1
        return output
# Inputs to the model
query = torch.randn(1, 64, 56, 56)
key = torch.randn(1, 64, 56, 56)
val = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
# The model end

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q0, k3, v2, mask):
        qk = q0 @ torch.transpose(k3, -2, -1) / math.sqrt(q0.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v2
        return output
# Inputs to the model
query = torch.randn(1, 64, 56, 56)
key = torch.randn(1, 64, 56, 56)
val = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
# Model end

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k4, v3, mask):
        qk = q @ torch.transpose(k4, -2, -1)
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v3
        return output
# The model inputs
q = torch.randn(1, 64, 56, 56)
k = torch.randn(1, 64, 56, 56)
v = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
# Model end

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q8, k5, v4, mask_v):
        qk = torch.transpose(q8, 0, 1) @ k5 / math.sqrt(q8.size(-1))
        qk = qk + mask_v
        attn_weight = torch.softmax(qk, dim=-1)
        output = torch.transpose(attn_weight, 0, 1) @ v4
        return output
# Inputs
q0 = torch.randn(64, 1, 56, 56)
k0 = torch.randn(64, 1, 56, 56)
v0 = torch.randn(64, 1, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
# Model end
