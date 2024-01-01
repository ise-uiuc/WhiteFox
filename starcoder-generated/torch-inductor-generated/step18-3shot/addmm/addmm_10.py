
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(inp, x1)
        v2 = v1 + x2.T
        v3 = v2.contiguous()
        return v1
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 3)
inp = torch.randn(4, 5)
