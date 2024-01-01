
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, 1, stride=1, padding=0),
            torch.nn.ReLU(),
        )
        self.conv1 = torch.nn.Conv2d(64, 32, 8, stride=2, padding=3)
        self.conv2 = torch.nn.Conv2d(32, 16, 16, stride=2, groups=4)
    def forward(self, x):
        y = self.layers(x)
        y = self.conv1(y)
        y = y + 0.3
        y = self.conv2(y)
        y = y - 0.4
        return y
# Inputs to the model
x = torch.randn(1, 128, 1, 1)
