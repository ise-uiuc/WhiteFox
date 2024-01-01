
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        k1: torch.Tensor = torch.Tensor(66, 66)
        k1.fill_(1.2)
        k2: torch.Tensor = torch.Tensor(66, 66)
        k2.fill_(0.8)
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x2 + k1)
        v2 = torch.mm(inp, v1 + k2)
        return v2
# Inputs to the model
x1 = torch.randn(66, 66)
x2 = torch.randn(66, 66)
inp = torch.randn(66, 66)
