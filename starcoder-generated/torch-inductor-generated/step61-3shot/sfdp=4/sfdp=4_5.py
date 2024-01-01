
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K, V, m2):
        qk = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        qk = qk + m2
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V
        return output
# Inputs to the model
Q = torch.randn(1, 3, 100, 100)
K = torch.randn(1, 3, 100, 100)
V = torch.randn(1, 3, 100, 100)
mask = (torch.rand(1, 100, 100) > 0.7).fill_(-1000000000.0)
