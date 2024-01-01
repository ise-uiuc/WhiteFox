
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp1, inp2, inp3):
        v1 = torch.mm(inp1, inp2)
        v2 = v1 + x1
        v3 = torch.mm(v2, x2)
        v4 = torch.mm(v3, v1)
        return v4
# Inputs to the model
x1 = torch.randn(3, 5)
x2 = torch.randn(5, 3)
inp1 = torch.randn(6, 2)
inp2 = torch.randn(2, 4)
inp3 = torch.randn(4, 6)
