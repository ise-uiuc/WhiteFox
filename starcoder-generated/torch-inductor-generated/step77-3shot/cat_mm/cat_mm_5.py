
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = []
        v += [torch.mm(x1, x2)] * 3
        v += [torch.mm(x1, x2)] * 5
        v += [torch.mm(x1, x2)] * 1
        return torch.cat(v, 1)
# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(1, 3)
