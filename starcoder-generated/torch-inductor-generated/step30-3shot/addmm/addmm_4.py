
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v = torch.mm(x1, x2.flatten())
        return v + inp
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 3)
inp = torch.randn(2, 3)
