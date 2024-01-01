
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.nn.Linear(3, 3)
        self.t1 = torch.nn.Tanh()

    def forward(self, x1, x2, inp):
        v1 = torch.mm(inp, inp)
        y = self.t1(self.m1(inp))
        v2 = v1 + y
        v2 = v2 + x1
        return torch.mm(v2, x2)
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
