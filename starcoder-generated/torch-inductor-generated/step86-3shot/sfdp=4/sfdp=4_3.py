
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q19, k1, v11, mask):
        qk = q19 @ k1.transpose(-2, -1) / math.sqrt(q19.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ v11
        return output
# Inputs to the model
q14 = torch.randn(1, 64, 56, 56)
k6 = torch.randn(1, 64, 56, 56)
v10 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
