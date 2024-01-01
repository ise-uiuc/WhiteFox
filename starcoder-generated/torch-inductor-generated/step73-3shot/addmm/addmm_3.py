
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(3, 4)
    def forward(self, x, inp):
        v1 = self.linear1(x) + inp
        v2 = self.linear2(v1)
        return self.linear2(v2)
# Inputs to the model
x = torch.randn(3, 2)
inp = torch.randn(3, 2)
