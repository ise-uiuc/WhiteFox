
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, query3, key10, value6, mask):
        qk = query3 @ key10.transpose(-2, -1) / math.sqrt(query3.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ value6
        return output
# Inputs to the model
Q10 = torch.randn(1, 64, 56, 56)
K1 = torch.randn(1, 64, 56, 56)
V3 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
