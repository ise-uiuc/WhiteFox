
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp, bias):
        v1 = torch.mm(inp, x1)
        return torch.add(v1, bias)
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3, requires_grad=True)
bias = torch.randn(3, 3, requires_grad=True)
