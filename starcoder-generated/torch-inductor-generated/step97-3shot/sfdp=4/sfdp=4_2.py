
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q3, K3, V10, mask):
        qk = Q3 @ K3.transpose(-2, -1) / math.sqrt(Q3.size(-1))
        qk = qk + mask
        attn_w = torch.softmax(qk, dim=-1)
        output = attn_w @ V10
        return output
# Inputs to the model
Q2 = torch.randn(1, 64, 56, 56)
K3 = torch.randn(1, 64, 56, 56)
V10 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
