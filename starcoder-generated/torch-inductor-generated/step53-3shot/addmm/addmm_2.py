
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, v0, inp):
        x1 = torch.mm(inp, inp)
        x2 = torch.mm(x1, torch.mm(inp, inp))
        x3 = x2.permute(1, 0)
        return torch.mm(x2, x3) + v0
# Inputs to the model
inp = torch.randn(8, 1, 3)
v0 = torch.randn(1, 8, 3, requires_grad=True)
