
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, inp)
        v2 = v1 + x1
        v3 = torch.mm(x2, v2)
        v4 = v1 + torch.mm(x2, inp)
        return v3 + v4
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
