
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 7, stride=2, padding=3)
        self.conv2 = torch.nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(256, 2, 3, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        return v5
# Inputs to the model
x = torch.randn(1, 16, 28, 28)
