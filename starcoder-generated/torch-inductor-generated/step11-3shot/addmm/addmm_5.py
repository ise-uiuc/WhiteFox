
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(inp, x2)
        v2 = v1 + x1
        return v2
# Inputs to the model
x1 = torch.randn(385, 17)
x2 = torch.randn(17, 1093)
inp = torch.randn(385, 1093)
