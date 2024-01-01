
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(7, 4), nn.ReLU(), nn.Linear(4, 4))
    def forward(self, x):
        x = self.layers(x.flatten(0, 1))
        x = x.flatten(start_dim=1)
        return x
# Inputs to the model
x = torch.randn(1, 1, 7)
