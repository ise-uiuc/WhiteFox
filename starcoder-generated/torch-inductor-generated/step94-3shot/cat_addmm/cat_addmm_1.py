
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Conv2d(32, 16, 1)
    def forward(self, x):
        x = self.layers(x)
        x = torch.cat((x, x), dim=1)
        x = torch.cat((x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(4, 32, 1, 1)
