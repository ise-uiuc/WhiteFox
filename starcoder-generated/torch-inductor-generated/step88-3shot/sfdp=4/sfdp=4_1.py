
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k2, v, mask):
        qk = q @ k2.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
# Inputs to the model
Q4 = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask = (torch.randint(2, (1, 56, 56)) > 0).fill_(-1000000000.0)
