
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax_a = torch.nn.Softmax(dim=-10000)
    def forward(self, q, k, v, m7):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + m7
        attn_weight = self.softmax_a(qk)
        output = attn_weight @ v
        return output
# Inputs to the model
Q = torch.randn(1, 56, 56, 64)
K = torch.randn(1, 56, 56, 64)
V = torch.randn(1, 56, 56, 64)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
