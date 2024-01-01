
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=1, dilation=1)
        self.avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.avgpool(v2)
        v4 = self.conv3(v3)
        v5 = torch.sigmoid(v4)
        v6 = self.conv4(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
