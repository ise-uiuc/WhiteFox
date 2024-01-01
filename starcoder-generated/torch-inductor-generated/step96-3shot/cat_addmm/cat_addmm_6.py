
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 2, 1)
        self.stack = torch.stack
    def forward(self, x):
        x = self.conv(x)
        x = torch.stack((x, x, x, x))
        x = self.stack((x, x))
        x = x.flatten(dim=0)
        return x
# Inputs to the model
x = torch.randn(1, 4, 5, 5)
