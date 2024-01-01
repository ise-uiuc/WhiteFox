
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    def forward(self, x):
        x = self.layers(x)
        x = x.flatten(start_dim=1)
        x = torch.stack((x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 4)
