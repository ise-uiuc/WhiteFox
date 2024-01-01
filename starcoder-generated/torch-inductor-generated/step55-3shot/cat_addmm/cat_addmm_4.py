
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.BatchNorm1d(16)
        self.layers2 = nn.Linear(8, 4)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x), dim=1).flatten(1)
        x = self.layers2(x)
        return x
# Inputs to the model
x = torch.randn(1, 16)
