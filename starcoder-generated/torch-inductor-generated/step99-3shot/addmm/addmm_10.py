
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x2, inp):
        x1 = torch.randn(3, 3)
        v1 = torch.mm(x1, self.inp)
        v2 = torch.mm(x2, inp)
        v1 = v1 + v2
        return v1
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
