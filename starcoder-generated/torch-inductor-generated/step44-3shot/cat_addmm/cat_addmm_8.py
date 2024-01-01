
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(16, 32)
        self.layers2 = nn.Linear(32, 2)
    def forward(self, x):
        x = self.layers(x)
        x = self.layers2(x)
        z = torch.stack((x, x), dim=2)
        y = torch.cat((z, z, z), dim=2)
        y = torch.flatten(y, 1)
        return y
# Inputs to the model
x = torch.randn(4, 16)
