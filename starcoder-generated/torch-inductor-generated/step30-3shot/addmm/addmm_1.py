
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(inp, x2)
        v2 = v1+ x1
        return v2* x2
# Inputs to the model
x1 = torch.randn(3, 3).cpu()
x2 = torch.randn(3).cpu()
inp = torch.randn(3, 3)
