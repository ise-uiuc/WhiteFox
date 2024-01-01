
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query2, key, value, mask):
        q = query2
        k = key
        v = value
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
# Inputs to the model
Q = torch.randn(1, 56, 64)
K = torch.randn(1, 64, 56)
V = torch.randn(1, 64, 56)
mask = (torch.rand(1, 64) > 0.7).fill_(-1000000000.0)
