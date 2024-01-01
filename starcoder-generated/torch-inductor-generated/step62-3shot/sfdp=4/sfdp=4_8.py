
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q2 = torch.nn.Linear(56*56, 8*56)
        self.k = torch.nn.Linear(8*56, 8*56)
    def forward(self, Q8, K2, V4, mask):
        qk = self.q2(Q8) @ self.k(K2).transpose(-2, -1) / math.sqrt(self.q2(Q8).size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ V4
        return output
# Inputs to the model
Q = torch.randn(1, 56*56)
K = torch.randn(1, 56*56)
V = torch.randn(1, 56*56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
