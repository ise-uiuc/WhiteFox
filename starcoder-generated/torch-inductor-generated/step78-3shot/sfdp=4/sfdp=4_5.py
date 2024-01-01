
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v1, mask):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v1
        return output
# Inputs to the model
Q5 = torch.randn(1, 64, 56, 56)
K6 = torch.randn(1, 64, 56, 56)
V2 = torch.randn(1, 64, 56, 56)
mask2 = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
