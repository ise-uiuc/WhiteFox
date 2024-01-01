
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v = torch.nn.Parameter(torch.randn(2, 3))
    def forward(self, x1, x2):
        v = []
        v += [torch.mm(x1, self.v)] * 3
        v += [torch.mm(x2, self.v)] * 3
        return torch.cat(v, 1)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
