
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, k, v, q):
        qk = q @ k.transpose(-2, -1)
        qk = qk / math.sqrt(q.size(-1))
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
# Inputs to the model
Q20 = torch.randn(1, 64, 56, 56)
k5 = torch.randn(1, 64, 56, 56)
V30 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
