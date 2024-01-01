
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        x1 = torch.randn(3, 3, requires_grad=True)
        x2 = torch.randn(3, 3, requires_grad=True)
        inp = torch.randn(3, 3, requires_grad=True)
    def forward(self, x1, x3, inp):
        v1 = torch.mm(inp, x1)
        v1 = torch.add(v1, x3)
        return v1
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x3 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3, requires_grad=True)
