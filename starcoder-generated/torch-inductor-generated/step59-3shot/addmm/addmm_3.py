
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, x2, inp):
        v1 = torch.mm(x, x)
        v2 = torch.mul(v1, x)
        v2 = v2 + v1
        v2 = v2 + x
        return torch.mm(v2, inp)
# Inputs to the model
x = torch.randn(4, 3, requires_grad=True)
x2 = torch.randn(3, 3)
inp = torch.randn(4, 6, requires_grad=True)
