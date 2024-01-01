
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.rand(1, 1, 64) * 10.
        self.k = torch.rand(1, 1, 64) * 10.
        self.v = torch.rand(1, 1, 64) * 10.
        self.mask = torch.rand(1, 56, 56) > 0.7

    def forward(self, q):
        qk = q @ self.q.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
# Inputs to the model
Q = torch.randn(1, 1, 64)
