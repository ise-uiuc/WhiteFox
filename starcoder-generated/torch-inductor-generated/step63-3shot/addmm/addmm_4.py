
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, inp=None):
        x3 = torch.add(x1, inp)
        x3 = torch.add(x3, inp)
        x4 = torch.add(x1, 3)
        x5 = torch.add(x3, 3)
        x4 = x4 * x5
        return x4 + inp * inp
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3)
