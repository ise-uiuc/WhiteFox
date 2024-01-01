
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp1, inp2):
        v1 = torch.mm(inp1, inp2)
        v2 = v1 + x1
        v3 = torch.mm(v2, x2)
        return v3
# Inputs to the model
x1 = torch.randn(3, 5)
x2 = torch.randn(5, 3)
inp1 = torch.randn(1, 1)
inp2 = torch.randn(1, 1)
