
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=5, out_channels=4, kernel_size=1, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(4)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=4)
        self.conv2 = torch.nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1, stride=1)
        self.bn2 = torch.nn.BatchNorm2d(3)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=4, stride=5)
        self.conv3 = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=1, stride=1)
        self.conv4 = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.sigmoid2 = torch.nn.Sigmoid()
        self.sigmoid3 = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = torch.sigmoid(v2)
        v4 = self.maxpool1(v3)
        v5 = self.conv2(v4)
        v6 = self.bn2(v5)
        v7 = torch.sigmoid(v6)
        v8 = self.maxpool2(v7)
        v9 = self.conv3(v8)
        v10 = torch.sigmoid(v9)
        v11 = self.conv4(v10)
        v12 = torch.sigmoid(v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 5, 15, 15)
