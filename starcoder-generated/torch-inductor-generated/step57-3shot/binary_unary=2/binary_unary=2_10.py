
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 16, 16, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 32, 1, stride=1, padding=0, dilation=1, groups=0)
        self.conv4 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0, dilation=2, groups=0)
        self.conv5 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0, dilation=4, groups=0)
        self.conv6 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0, dilation=1, groups=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.pool1(v2)
        v4 = self.conv3(v3)
        v5 = self.conv4(v4)
        v6 = self.conv5(v5)
        v7 = self.conv6(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 362, 362)
