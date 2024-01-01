
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 2)
    def forward(self, x1, x2):
        x = torch.mm(x1, x2)
        v1 = self.layer(x)
        return torch.cat([v1, v1, v1], 1)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
