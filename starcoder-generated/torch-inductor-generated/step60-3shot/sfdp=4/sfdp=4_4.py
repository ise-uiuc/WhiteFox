
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q1, K, V, mask3):
        qk = Q1 @ K.transpose(-2, -1) / math.sqrt(Q1.size(-1))
        qk = qk + mask3
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V
        return output
# Inputs to the model
Q1 = torch.randn(1, 32, 24, 24)
K = torch.randn(1, 32, 24, 24)
V = torch.randn(1, 32, 24, 24)
mask3 = (torch.rand(1, 24, 24) > 0.7).fill_(-1000000000.0)
