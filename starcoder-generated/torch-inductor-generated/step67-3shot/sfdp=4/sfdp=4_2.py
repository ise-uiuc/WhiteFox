
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q1, k1, v1, mask1):
        qk = q1 @ k1.transpose(-2, -1) / math.sqrt(q1.size(-1))
        qk = qk + mask1
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v1
        return output
# Inputs to the model
Q = torch.randn(1, 512)
K = torch.randn(1, 512)
V = torch.randn(1, 512)
mask = (torch.rand(1, 512) > 0.7).fill_(-1000000000.0)
