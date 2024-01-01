
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, qn, n, m, mask):
        qn = qn @ n.transpose(-2, -1) / math.sqrt(qn.size(-1))
        qn = qn + m
        attn_weight = torch.softmax(qn, -1)
        output = attn_weight @ mask
        return output
# Inputs to the model
Q4 = torch.randn(1, 64, 56, 56)
N = torch.randn(1, 64, 56, 56)
m = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
mask = torch.randn(1, 64, 56, 56)
