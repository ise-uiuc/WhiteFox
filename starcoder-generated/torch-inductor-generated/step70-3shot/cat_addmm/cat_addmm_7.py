
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Conv2d(3, 2, 2)
    def forward(self, x):
        x = self.layers(x)
        x = torch.cat((x, x, x, x, x), dim=1).flatten(0)
        return x
# Inputs to the model
x = torch.randn(1, 3, 3, 3)
