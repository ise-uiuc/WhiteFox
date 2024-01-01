
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k1, v8, mask):
        qk = q @ k1.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v8
        return output
# Inputs to the model
Q17 = torch.randn(1, 64, 56, 56)
K1 = torch.randn(1, 64, 56, 56)
V8 = torch.randn(1, 64, 56, 56)
mask11 = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
