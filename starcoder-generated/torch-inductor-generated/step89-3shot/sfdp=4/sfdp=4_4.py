
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q7, k0, v, bias):
        qk = q7 @ k0.transpose(-2, -1) / math.sqrt(q7.size(-1))
        qk = qk + bias
        output = (torch.softmax(qk, dim=-1)) @ v
        return output
# Inputs to the model
qq = torch.randn(1, 64, 56, 56)
k1 = torch.randn(1, 64, 56, 56)
V12 = torch.randn(1, 384, 384)
bias1 = (torch.rand(1, 56, 56) > 0.7).fill_(-1000000000.0)
