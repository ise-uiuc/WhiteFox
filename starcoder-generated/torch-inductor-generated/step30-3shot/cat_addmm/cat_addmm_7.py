
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(6, 4), nn.ReLU(), nn.Flatten(start_dim=1), nn.Linear(4, 4))
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack([x, x], dim=1)
        x = x.flatten(start_dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 6)
