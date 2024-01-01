
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp, x5):
        v1 = torch.mm(inp, x2)
        v2 = v1 + x2
        v3 = torch.mm(inp, x2)
        return v2 + x5
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3, requires_grad=True)
x5 = torch.randn(1, 3)
