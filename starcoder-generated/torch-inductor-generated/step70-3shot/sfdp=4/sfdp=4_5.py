
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1, 64, 56, 56), requires_grad=False)
    def forward(self, q, k, v, mask):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ self.weight
        return output
# Inputs to the model
Q3 = torch.randn(1, 64, 56, 56)
K4 = torch.randn(1, 64, 56, 56)
V1 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
