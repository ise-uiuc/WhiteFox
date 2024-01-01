
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        x = torch.cat([x, x, x], dim=1)
        x = x.relu()
        y = torch.cat([y, y, y], dim=-1)
        y = y.relu()
        return x, y
# Inputs to the model
x = torch.randn(2, 3, 4)
y = torch.randn(2, 3, 4)
