
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q1, ks, k, v, mask1):
        qk = q1 @ k.transpose(-2, -1) / math.sqrt(q1.size(-1))
        qk = qk + mask1
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
# Inputs to the model
Q1 = torch.randn(1, 64, 56, 56)
K = torch.randn(1, 64, 56, 56)
V = torch.randn(1, 64, 56, 56)
mask2 = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
