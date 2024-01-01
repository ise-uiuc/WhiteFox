
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp1, inp2):
        v0 = torch.mm(inp1, inp1)
        v1 = torch.mm(inp2, inp2)
        v2 = v0 + x1
        v3 = v1 + x1
        return v2 + v3
# Inputs to the model
x1 = torch.randn(3, 5)
x2 = torch.randn(5, 5)
inp1 = torch.randn(5, 5)
inp2 = torch.randn(3, 5)
