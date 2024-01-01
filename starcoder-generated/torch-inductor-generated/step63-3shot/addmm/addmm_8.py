
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, inp):
        x2 = x1 + inp
        x2 = torch.mm(x2, x1)
        x2 = x2 + inp
        x2 = torch.mm(x2, x1)
        v1 = x1 + x2
        return torch.mm(v1, inp)
# Inputs to the model
x1 = torch.randn(3, requires_grad=True)
inp = torch.randn(3, 3)
