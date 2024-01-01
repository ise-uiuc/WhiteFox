
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 3)
    def forward(self, x1, x2, inp):
        v1 = torch.mm(self.linear1(x1), self.linear1(x2))
        v2 = v1 + inp
        return v2
# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(2, 3)
inp = torch.randn(2, 3)
