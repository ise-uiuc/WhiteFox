
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Conv2d(1, 2, 3)
    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 1)
        x = torch.stack((x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 1, 3, 3)
