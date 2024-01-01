
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q8, K8, V9, mask):
        qk = Q8 @ K8.transpose(-2, -1) / math.sqrt(Q8.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V9
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K5 = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 128, 16, 16)
mask = (torch.rand(1, 16, 16) > 0.7).fill_(-1000000000.0)
