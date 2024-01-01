
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, inp):
        x2 = torch.mm(x1, x1)
        v1 = torch.mm(x1, x2)
        v2 = v1 + x1
        v3 = torch.mm(x1, x2)
        return v3 + inp
# Inputs to the model
x1 = torch.randn(3, 3)
inp = torch.randn(3, 3)
