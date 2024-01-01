
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 3), nn.Linear(3, 4)])
        self.flatten = nn.Flatten()
        self.bn = nn.BatchNorm1d(4)
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        x = torch.stack([x, x], dim=1)
        x = self.flatten(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
