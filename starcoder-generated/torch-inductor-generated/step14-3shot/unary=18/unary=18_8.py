
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=9, padding=5, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=8, padding=5, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, padding=5, stride=2)
        self.avgpool = torch.nn.AvgPool2d(7)
        self.maxpool = torch.nn.MaxPool2d(3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.avgpool(v6)
        v8 = torch.sigmoid(v7)
        v9 = self.maxpool(v6)
        v10 = torch.sigmoid(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 1, 96, 96)
