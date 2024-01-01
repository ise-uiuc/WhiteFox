
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3, bias=False)
    def forward(self, x1, x2, inp):
        v1 = self.linear(x2)
        v2 = torch.mm(v1, x1)
        return torch.mm(v2, x2 + x1)
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
