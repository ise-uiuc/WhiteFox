
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, inp):
        v1 = torch.mm(x, x)
        v2 = torch.mm(v1, v1)
        v3 = v2
        return v1 * v3 + inp
# Inputs to the model
x = torch.randn(3, 3)
inp = torch.randn(3, 3)
