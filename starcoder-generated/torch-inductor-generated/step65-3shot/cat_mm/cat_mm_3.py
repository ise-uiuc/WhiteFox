
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 3)
        self.linear2 = torch.nn.Linear(3, 2)
    def forward(self, x1, x2):
        v1 = self.linear1(x1)
        v2 = self.linear2(v1)
        return torch.cat([x1, v1, v2], 1)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 3)
