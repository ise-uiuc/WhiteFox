
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(inp, x1)
        v2 = v1 + x2
        return v2
# Inputs to the model
x1 = torch.randn(5, 20)
x2 = torch.randn(5, 5)
inp = torch.randn(20, 5)
