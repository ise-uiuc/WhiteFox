
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 7, stride=2, padding=3, groups=4)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1, groups=4)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=4)
        self.conv4 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=2, groups=4)
        self.conv5 = torch.nn.Conv2d(64, 64, 1, stride=2, padding=0, groups=4)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 16, 512, 512)
