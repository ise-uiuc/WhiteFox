
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, inp1, inp2, inp3, inp4):
        v1 = torch.mm(inp1, inp2)
        v2 = torch.mm(inp3, inp4)
        v3 = v1 + x1
        v4 = v2 + x2
        v5 = v3 * x3
        v6 = v4 * x4
        return v5 + v6
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3)
x4 = torch.randn(3, 3)
inp1 = torch.randn(3, 3)
inp2 = torch.randn(3, 3)
inp3 = torch.randn(3, 3)
inp4 = torch.randn(3, 3)
