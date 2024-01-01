
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k, v, mask):
        qk = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v
        return output
# Inputs to the model
q = torch.randn(1, 64, 8, 8)
k = torch.randn(1, 16, 15, 15)
v = torch.randn(1, 128, 15, 15)
mask = (torch.rand(1, 15, 15) > 0.7).fill_(-1000000000.0)
