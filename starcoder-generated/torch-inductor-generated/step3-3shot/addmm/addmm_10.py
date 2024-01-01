
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        t = inp.transpose(-2, -1)
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(v1, t)
        return v2
# Inputs to the model
x1 = torch.randn(6, 12)
x2 = torch.randn(6, 6)
inp = torch.randn(12, 6)
