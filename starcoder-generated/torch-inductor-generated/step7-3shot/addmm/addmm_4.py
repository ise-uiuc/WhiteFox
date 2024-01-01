
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x2)
        v2 = v1 + inp
        return v2
# Inputs to the model
x1 = torch.randn(2, 3, 4, 5, 6, 6, 7, 8)
x2 = torch.randn(8, 7, 6, 5, 4, 5, 8, 3)
inp = torch.randn(2, 3, 4, 5, 2, 8, 18, 38)
