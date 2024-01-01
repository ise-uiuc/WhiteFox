
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q15, KKK7, VVV2, mask):
        qk = Q15 @ KKK7.transpose(-2, -1) / math.sqrt(Q15.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ VVV2
        return output
# Inputs to the model
Q8 = torch.randn(1, 64, 56, 56)
KKK9 = torch.randn(1, 64, 56, 56)
VVV6 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
