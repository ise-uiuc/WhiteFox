
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, inp):
        v2 = torch.mm(x, x)
        v1 = v2 + inp
        return v1
# Inputs to the model
x = torch.randn(3, 3)
inp = torch.randn(3, 3)
