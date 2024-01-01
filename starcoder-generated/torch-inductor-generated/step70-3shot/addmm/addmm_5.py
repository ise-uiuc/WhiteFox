
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 3, requires_grad=False)
        self.linear2 = torch.nn.Linear(3, 3, requires_grad=True)
    def forward(self, x1, x2, inp):
        v1 = self.linear1(x1)
        v2 = self.linear2(inp)
        v3 = v1 + v2
        v4 = self.linear1(v3) + inp
        v5 = self.linear1(v4) + 1
        return v1 + v2
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=False)
inp = torch.randn(3, 3, requires_grad=False)
