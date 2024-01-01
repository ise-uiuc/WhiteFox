
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q3, K2, V, mask):
        qk = Q @ K2.transpose(-2, -1) / math.sqrt(Q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, d = -1)
        output = attn_weight @ V
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V4 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
