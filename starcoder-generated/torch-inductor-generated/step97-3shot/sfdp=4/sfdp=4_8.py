
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q5, K7, V1, mask):
        qk = Q5 @ K7.transpose(-2, -1) / math.sqrt(Q5.size(-1))
        qk = qk + mask
        attn_w = torch.softmax(qk, dim=-1)
        output = attn_w @ V1
        return output
# Inputs to the model
Q5 = torch.randn(1, 64, 56, 56)
K7 = torch.randn(1, 64, 56, 56)
V1 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
