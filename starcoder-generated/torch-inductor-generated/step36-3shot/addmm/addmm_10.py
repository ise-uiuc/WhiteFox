
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, inp):
        v1 = torch.mm(x1, x2)
        v1 = x3 + torch.mm(v1, x4)
        v1 = torch.mm(v1, inp)
        return v1
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3)
x4 = torch.randn(3, 3)
inp = torch.randn(3, 3)
