
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(32, 64, stride=1, padding=0)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.avgpool(v3)
        v5 = self.conv4(v4)
        v6 = self.conv5(v5)
        return v6
# Inputs to the model
x1 = torch.randn(2, 3, 28, 28)
