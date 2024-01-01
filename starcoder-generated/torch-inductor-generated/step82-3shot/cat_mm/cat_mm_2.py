
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = self.lin(x1)
        v2 = self.lin(x1)
        return torch.cat([v1, v1, v1, v1], 1)
# Inputs to the model
x1 = torch.randn(2, 2)
