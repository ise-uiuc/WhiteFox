
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp, inp2):
        v1 = torch.mm(inp2, x2 + 323)
        v2 = v1 - inp
        return v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = 1
inp2 = torch.randn(3, 3)
