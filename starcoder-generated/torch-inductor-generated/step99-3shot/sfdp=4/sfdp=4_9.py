
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k4, v, mask30):
        qk = q @ k4.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask30
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
# Inputs to the model
Q = torch.randn(1, 1024, 29, 29)
K = torch.randn(1, 1024, 29, 29)
V = torch.randn(1, 1024, 29, 29)
mask = (torch.rand(1, 29, 29) > 0.7).fill_(-1000000000.0)
