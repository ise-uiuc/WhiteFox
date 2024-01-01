
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K1, V, mask):
        qk = Q @ K1.transpose(-2, -1) / math.sqrt(Q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V
        return output
# Inputs to the model
Q = torch.randn(1,64, 56, 56)
K = torch.randn(1,64, 27, 27)
V = torch.randn(1,64, 56, 56)
mask = (torch.rand(1, 27, 27) > 0.8).fill_(-1000000000.0)
