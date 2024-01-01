
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q0, k0, v0, v1, mask):
        qk = q0 @ k0.transpose(-2, -1) / math.sqrt(q0.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v0
        output = output @ v1
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 128)
K = torch.randn(1, 64, 128, 56)
V0 = torch.randn(1, 64, 56, 128)
V1 = torch.randn(1, 64, 128, 56)
mask = (torch.rand(1, 56, 128) > 0.7).fill_(-1000000000.0)
