
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,  Q3, K1, V4, mask):
        qk = Q3 @ K1.transpose(-2, -1) / math.sqrt(Q3.size(-1))
        qk = qk + mask
        attn_weigt = torch.softmax(qk, dim=-1)
        output = attn_weigt @ V4
        return output
# Inputs to the model
Q3 = torch.randn(1, 64, 56, 56)
K1 = torch.randn(1, 64, 56, 56)
V4 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
