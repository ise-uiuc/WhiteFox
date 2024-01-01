
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, k6, v6, mask6):
        qk = x @ k6.transpose(-2, -1) / math.sqrt(x.size(-1))
        qk = qk + mask6
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v6
        return output
# Inputs to the model
x5 = torch.randn(1, 64, 56, 56)
k6 = torch.randn(1, 64, 56, 56)
v6 = torch.randn(1, 64, 56, 56)
mask6 = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
