
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x3, inp):
        v1 = torch.mm(x1, x2)
        v3 = torch.sigmoid(v1)
        v2 = v3 + inp
        return v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3, requires_grad=True)
x3 = torch.randn(10, 3, 3)
inp = torch.randn(10, 3, 3, requires_grad=True)
