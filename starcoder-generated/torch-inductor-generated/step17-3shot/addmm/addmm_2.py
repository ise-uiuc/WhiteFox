
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x2)
        v2 = v1 + inp
        return v2
# Inputs to the model
x1 = torch.randn(4, 5)
x2 = torch.randn(5, 6, 7)
inp = torch.randn(10, 6, 7)
