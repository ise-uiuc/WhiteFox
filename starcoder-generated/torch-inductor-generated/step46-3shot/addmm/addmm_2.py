
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(self)
    def forward(self, x1, x2, x3, inp):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x3, x2) + v1
        return v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3, requires_grad=True)
x3 = torch.randn(3, 3)
inp = torch.randn(3, 3, requires_grad=True)
