
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, inp)
        v2 = torch.add(x1, v1)
        return torch.mul(v1, v2)
# Inputs to the model
x1 = torch.randn(3, 1)
x2 = torch.randn(3, 1, requires_grad=True)
inp = torch.randn(1, 3)
