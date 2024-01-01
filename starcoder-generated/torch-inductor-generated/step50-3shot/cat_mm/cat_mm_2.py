
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l4 = torch.nn.Linear(2, 2)
    def forward(self, x1, x2):
        v1 = self.l4(x2)
        v2 = torch.mm(x1, v1)
        v3 = torch.mm(x1, v1)
        v4 = torch.mm(x1, v1)
        return torch.cat([v1, v2, v3, v4], 1)
# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(1, 2)
