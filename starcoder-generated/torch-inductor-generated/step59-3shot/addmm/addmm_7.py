
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp, x3=3):
        v1 = torch.mm(inp, x2)
        x3 = x3 - 1
        v2 = v1 + float(x3)
        return torch.addmm(v2, inp, x1)
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3, requires_grad=True)
