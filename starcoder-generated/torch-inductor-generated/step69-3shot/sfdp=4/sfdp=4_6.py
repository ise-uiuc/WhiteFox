
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query, key, value, attn_mask):
        qk = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        qk = qk + 0.33 * attn_mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value
        return output
# Inputs to the model
Q = torch.randn(1, 64, 256)
K = torch.randn(1, 64, 256)
V = torch.randn(1, 64, 256)
mask = (torch.rand(1, 256) > 0.7).fill_(-1000000000.0)
mask = torch.stack([mask, mask, mask], dim=-1).unsqueeze(1)
# model ends

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, V, mask):
        qk = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V
        return output
# Inputs to the model
Q = torch.randn(1, 16, 56, 56)
K = torch.randn(1, 16, 56, 56)
V = torch.randn(1, 16, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
