
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, Q7, K4, V4, mask):
        qk = Q7 @ K4.transpose(-2, -1) / math.sqrt(Q7.size(-1))
        qk = qk + mask
        att_weight = torch.softmax(qk, dim=-1)
        output = att_weight @ V4
        return output
# Inputs to the model
Q7 = torch.randn(1, 64, 56, 56)
K4 = torch.randn(1, 64, 56, 56)
V4 = torch.randn(1, 64, 56, 56)
mask = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
