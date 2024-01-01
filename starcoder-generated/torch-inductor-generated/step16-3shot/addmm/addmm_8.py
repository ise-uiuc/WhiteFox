
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp1, inp2):
        v1 = torch.mm(inp2 + 323, x2)
        v2 = v1 + x1
        return v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(555, 3)
inp1 = torch.randn(3, 555)
inp2 = torch.randn(3, 555)
