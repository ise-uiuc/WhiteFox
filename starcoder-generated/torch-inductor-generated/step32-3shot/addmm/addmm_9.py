
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3, bias=False)
    def forward(self, x1, x2, inp):
        v1 = self.linear(x1)
        v2 = v1 + x1
        return torch.mm(x1, v2)
# Inputs to the model
x1 = torch.randn(3, 3) # random
x2 = torch.randn(3, 3) # random
inp = torch.randn(3, 3)
