
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        x3 = torch.mm(x1, x2)
        v1 = torch.mm(x1, x3)
        v2 = v1 + x1
        v3 = v2 + x2
        return v3
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
