
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, inp):
        x2 = torch.mm(x1, x1)
        v1 = torch.mm(x1, x2)
        v2 = v1 + inp
        v3 = torch.mm(x1, x2)
        return v2 + x2
# Inputs to the model
x1 = torch.randn(3, 3)
inp = torch.randn(3, 3)
