
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x2)
        a1 = v1.detach()
        z1 = torch.add(a1, x2)
        return torch.mm(z1, inp)
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3, requires_grad=True)
