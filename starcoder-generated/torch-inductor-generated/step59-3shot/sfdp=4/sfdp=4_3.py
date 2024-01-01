
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, W, V, Q6, M0):
        qk = W @ V.transpose(-2, -1) / math.sqrt(Q6.size(-1))
        qk = qk + M0
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ Q6
        return output
# Inputs to the model
Q = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
