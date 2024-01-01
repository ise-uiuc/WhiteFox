
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x2, x1)
        v2 = v1 + inp
        return v2
# Inputs to the model
x1 = torch.randn(1, 197)
x2 = torch.randn(113, 197)
inp = torch.randn(1, 113)
