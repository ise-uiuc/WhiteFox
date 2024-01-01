
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 64, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(128, 128, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v2)
        v5 = v4 + v3
        v6 = self.conv5(v5)
        return v6
x = torch.randn(1, 32, 64, 64)
