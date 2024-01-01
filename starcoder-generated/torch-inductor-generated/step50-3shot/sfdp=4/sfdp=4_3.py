
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q3, k, v, mask1):
        qk = Q3 @ k.transpose(-2, -1) / math.sqrt(Q3.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
# Inputs to the model
Q = torch.randn(1, 2, 3, 4)
K = torch.randn(1, 2, 3, 4)
V = torch.randn(1, 2, 3, 4)
mask = (torch.rand(1, 3, 4) > 0.7).fill_(-1000000000.0)
