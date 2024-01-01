
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q, K1, V7, mask):
        qk = Q8 @ KKK9.transpose(-2, -1) / math.sqrt(Q8.size(-1))
        qk = qk + mask
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ VVV6
        return output
# Inputs to the model
Q8 = torch.randn(1, 64, 56, 56)
KKK2 = torch.randn(1, 64, 56, 56)
VVV9 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
