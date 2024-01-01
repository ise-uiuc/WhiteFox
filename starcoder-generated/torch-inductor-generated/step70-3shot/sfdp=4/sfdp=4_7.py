
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, K, v1, mask):
        qk = q @ K.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v1
        return output
# Inputs to the model
Q4 = torch.randn(1, 64, 56, 56)
K4 = torch.randn(1, 64, 56, 56)
V1 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
