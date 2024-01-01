
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v0 = torch.transpose(inp, 0, 1)
        v1 = torch.mm(x2, v0)
        v2 = v1 + x1
        v3 = torch.mm(inp, x2)
        v4 = v2 + v3
        return v4
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
