
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v2, mask):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v2
        return output
# Inputs to the model
Q = torch.randn(3, 40, 128, 128)
K = torch.randn(3, 40, 128, 128)
V = torch.randn(3, 40, 128, 128)
mask = (torch.rand(3, 128, 128) > 0.7).fill_(-1000000000.0)
