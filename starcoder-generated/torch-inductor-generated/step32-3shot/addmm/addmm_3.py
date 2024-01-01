
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 3)
        self.linear2 = torch.nn.Linear(3, 3)
    def forward(self, x, inp):
        v1 = self.linear1(x)
        v2 = self.linear2(inp)
        return v1 + v2
# Inputs to the model
x = torch.randn(3, 3)
inp = torch.randn(3, 3)
