
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        x1 = x1 + inp
        v1 = torch.mm(x1, x2) + x1
        v2 = torch.mm(x1, x2) + torch.mm(x1, x2)
        return v2 + torch.mm(x2, x1)
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
