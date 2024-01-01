
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 64, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=(0, 1))
        self.conv3 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=(1, 0))
        self.conv4 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=(1, 1))
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = (x1 - x4).abs().mean().view(1)
        return x5
# Inputs to the model
x1 = torch.randn(1, 16, 24, 24)
