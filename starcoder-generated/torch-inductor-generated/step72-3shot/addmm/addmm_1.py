
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm1 = torch.nn.Linear(3, 4)
        self.mm2 = torch.nn.Linear(4, 2)
    def forward(self, x1, x2, inp):
        x1 = self.mm1(inp)
        x2 = torch.mm(x1, inp)
        return x2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
