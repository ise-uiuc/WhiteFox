
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x, x2, x3, m5):
        qk = x @ x2.transpose(-2, -1) / math.sqrt(x.size(-1))
        qk = qk + m5
        attn_weight = torch.softmax(qk, dim=-1)
        output = attn_weight @ x3
        return output
# Inputs to the model
x = torch.randn(1, 64, 56, 56)
x2 = torch.randn(1, 64, 56, 56)
x3 = torch.randn(1, 64, 56, 56)
m5 = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
