
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x] + [x for _ in range(3)], dim=1)
        y = y.view(-1)
        y = y.relu()
        return torch.stack([y, y])
# Inputs to the model
x = torch.randn(2, 3, 4)
