
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 6))
    def forward(self, x):
        x = self.layers(x)
        x = torch.cat([x, x], dim=1)
        x = x.flatten(start_dim=1)
        return x
# Inputs to the model
x = torch.randn(6, 2)
