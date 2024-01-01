
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        x1 = torch.t(x1)
        v1 = torch.bmm(x1, x2)
        v2 = v1
        v2 = v2 + inp
        return v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 5, 5)
inp = 1
