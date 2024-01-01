
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, y1, inp):
        v1 = torch.mm(x1, x2)
        z = torch.mm(y1, v1) + inp
        s = z.view(10)
        v2 = torch.mm(x1, x2) + s
        return v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
y1 = torch.randn(3, 3)
inp = torch.randn(10)

