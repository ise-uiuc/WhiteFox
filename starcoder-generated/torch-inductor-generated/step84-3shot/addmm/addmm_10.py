
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp, b1):
        v1 = torch.mm(inp, x1)
        v2 = torch.mm(b1, x2)
        return v1 + v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
b1 = torch.randn(3, 3)
