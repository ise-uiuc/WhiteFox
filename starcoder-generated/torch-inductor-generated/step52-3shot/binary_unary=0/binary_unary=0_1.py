
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.avgpool = torch.nn.AvgPool2d(11, stride=10)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv5 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv6 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv7 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.avgpool(v3)
        v5 = self.conv4(v4)
        v6 = self.conv5(v5)
        v7 = self.conv6(v6)
        v8 = self.conv7(v7)
        return v8
# Input to the model
x = torch.randn(1, 64, 64, 64)
#Model ends



