
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3, 3)
        self.linear2 = torch.nn.Linear(1, 2)
    def forward(self, x1):
        s = self.linear1(x1)
        t = self.linear1(x1)
        y = self.linear2(t)
        return (s, y)
# Inputs to the model
x1 = torch.rand((2, 3))
