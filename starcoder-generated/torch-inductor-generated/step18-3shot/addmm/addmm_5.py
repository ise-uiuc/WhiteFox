
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x2)
        t1 = v1[0] + torch.mean(inp, 1)
        return t1
# Inputs to the model
x1 = torch.randn(3, 10)
x2 = torch.randn(10, 2)
inp = torch.randn(10)
