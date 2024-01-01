
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x2)
        v2 = v1 + inp
        return v2
# Inputs to the model
x1 = torch.randn(0, 1, 2)
x2 = torch.randn(1, 2, 0)
inp = torch.randn(0, 2, 0)
