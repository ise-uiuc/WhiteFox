
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 3, bias=False)
        self.linear2 = torch.nn.Linear(3, 1, bias=False)
    def forward(self, x1, x2, inp):
        v1 = self.linear2(torch.mm(self.linear1(x1), x2))
        v2 = v1 + inp
        return torch.mm(v2, inp)
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3, 3)
