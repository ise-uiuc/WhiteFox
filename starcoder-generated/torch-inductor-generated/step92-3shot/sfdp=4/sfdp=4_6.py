
class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q10, K10, mask10):
        qk = Q10 @ K10.transpose(-2, -1) / math.sqrt(Q10.size(-1))
        qk = qk + mask10
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V10
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
