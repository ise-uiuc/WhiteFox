
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q56, K78, V, mask):
        qk = Q56 @ K78.transpose(-2, -1) / math.sqrt(Q56.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
