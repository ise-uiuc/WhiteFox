
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v, mask):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask.unsqueeze(-2)
        attn_weight = torch.softmax(qk, -1)
        output = attn_weight @ v
        return output
# Inputs to the model
Q5 = torch.randn(1, 64, 56, 56)
K4 = torch.randn(1, 64, 56, 56)
V4 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 1, 56, 56) > 0.7).fill_(-1000000000.0)
