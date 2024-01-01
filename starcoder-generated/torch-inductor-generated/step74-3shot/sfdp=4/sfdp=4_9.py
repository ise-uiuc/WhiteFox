
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k5, v1, msk):
        qk = q @ k5.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + msk
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v1
        return output
# Inputs to the model
Q1 = torch.randn(1, 64, 56, 56)
k5 = torch.randn(1, 64, 56, 56)
v1 = torch.randn(1, 64, 56, 56)
msk = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
