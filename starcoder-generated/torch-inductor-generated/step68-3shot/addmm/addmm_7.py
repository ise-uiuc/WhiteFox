
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        s = torch.randn(3, 3, requires_grad=True)
    def forward(self, x1, x2, inp):
        r = torch.randn(3, 3, requires_grad=True)
        v1 = torch.mm(x1, x2)
        x2 = inp
        v2 = v1 + x2
        x2 = x2 + v2
        v2 = v2 + x2
        return v1, v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3)
