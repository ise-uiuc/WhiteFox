
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 5), nn.Linear(10, 2))
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x, x, x), dim=1)
        x = torch.cat((x, x, x, x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 1)
