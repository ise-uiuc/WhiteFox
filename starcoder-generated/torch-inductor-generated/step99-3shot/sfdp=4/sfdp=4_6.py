
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, k22, v0, mask):
        qk = x1 @ k22.transpose(-2, -1) / math.sqrt(x1.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v0
        return output
# Inputs to the model
q = torch.randn(1, 64, 56, 56)
key = torch.randn(1, 64, 56, 56)
value = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
