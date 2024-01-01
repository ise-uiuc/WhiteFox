
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, inp):
        v1 = torch.mm(x2, x3)
        x1 = x3
        x1 = x2
        return torch.cat((v1, x1, inp), 0)
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3, requires_grad=True)
