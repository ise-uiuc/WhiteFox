
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp1, inp2):
        v1 = torch.mm(inp1, inp1)
        v2 = torch.mm(inp2, inp2)
        v3 = v1 + x1
        v4 = v2 + x2
        return v3 + v4
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp1 = torch.randn(3, 3)
inp2 = torch.randn(3, 3)
