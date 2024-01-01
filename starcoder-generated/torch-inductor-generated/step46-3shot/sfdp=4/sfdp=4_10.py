
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q1, k1, v1, mask):
        qk = q1 @ k1.transpose(-2, -1) / math.sqrt(q1.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v1
        return output
# Inputs to the model
q = torch.randn(1, 64, 56, 56)
k = torch.randn(1, 64, 56, 56)
v = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
