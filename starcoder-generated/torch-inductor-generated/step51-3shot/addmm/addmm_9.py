
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        if x1 is not None:
            v1 = torch.mm(x1, x2)
            return v1 + inp
        else:
            v1 = torch.mm(x1, x2)
            return v1 + x1
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
