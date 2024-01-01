
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q3, K3, V4, mask66):
        qk = Q3 @ K3.transpose(-2, -1) / math.sqrt(Q3.size(-1))
        qk = qk + mask66
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V4
        return output
# Inputs to the model
q3 = torch.randn(1, 64, 56, 56)
k3 = torch.randn(1, 64, 56, 56)
v4 = torch.randn(1, 64, 56, 56)
mask66 = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
