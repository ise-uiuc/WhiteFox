
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp1, inp2):
        v1 = x2 * inp1
        v2 = torch.mm(x1, v1)
        v3 = inp2 * v2
        v4 = x1 + v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 0)
x2 = torch.randn(0, 222)
inp1 = torch.randn(1, 1)
inp2 = torch.randn(1, 1)
