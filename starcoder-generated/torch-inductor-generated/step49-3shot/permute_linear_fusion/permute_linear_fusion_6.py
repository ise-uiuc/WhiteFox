
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = torch.nn.Linear(2, 120)
        self.linear1 = torch.nn.Linear(120, 84)
        self.linear2 = torch.nn.Linear(84, 10)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = self.linear0(x1)
        v3 = self.linear1(v2)
        v4 = self.linear2(v2)
        return x1
# Inputs to the model
x1 = torch.randn(2, 2, 2)
