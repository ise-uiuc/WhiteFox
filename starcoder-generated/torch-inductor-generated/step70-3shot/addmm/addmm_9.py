
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 3, False)
    def forward(self, x1, x2, inp):
        v1 = self.linear1(x1)
        v2 = v1.mm(inp)
        v3 = self.linear1(v2)
        return v3
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3, requires_grad=True)
