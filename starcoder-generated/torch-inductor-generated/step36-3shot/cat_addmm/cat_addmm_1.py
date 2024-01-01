
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, 2), nn.Linear(2, 2), nn.Softmax())
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x), dim=1)
        x = x.flatten(start_dim=1)
        return x
# Inputs to the model
x = torch.randn(4, 4)
