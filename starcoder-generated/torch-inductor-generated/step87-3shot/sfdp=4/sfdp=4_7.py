
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, v1, k3, q2, mask):
        qk = v1 @ k3.transpose(-2, -1) / math.sqrt(v1.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ q2
        return output
# Inputs to the model
Q2 = torch.randn(1, 64, 56, 56)
K3 = torch.randn(1, 64, 56, 56)
V4 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
