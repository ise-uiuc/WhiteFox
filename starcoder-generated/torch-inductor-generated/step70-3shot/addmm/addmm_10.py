
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 3, False)
        self.linear2 = torch.nn.Linear(3, 3, False)
    def forward(self, x1, x2, inp):
        v1 = self.linear1(inp)
        v2 = v1 + x1
        v3 = self.linear2(v2)
        v4 = v3.matmul(inp)
        return v4
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3, requires_grad=True)
