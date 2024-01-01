
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp, x1, x2):
        v1 = torch.mm(inp, x1)
        v1 = torch.mm(inp, x2)
        v1 = torch.mm(x1, v1)
        v1 = torch.mm(x1, v1)
        return v1 + x1
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3)
