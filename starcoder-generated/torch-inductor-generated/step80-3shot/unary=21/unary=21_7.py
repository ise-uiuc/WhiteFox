
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 1)
        self.conv_bn = torch.nn.BatchNorm2d(64)
        self.conv1 = torch.nn.Conv2d(64, 32, 1)
        self.conv1_bn = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 16, 1)
        self.conv2_bn = torch.nn.BatchNorm2d(16)
        self.conv3 = torch.nn.Conv2d(16, 8, 1)
        self.conv3_bn = torch.nn.BatchNorm2d(8)
        self.conv4 = torch.nn.Conv2d(8, 2, 1)
        self.conv4_bn = torch.nn.BatchNorm2d(2)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv_bn(v1)
        v3 = self.conv1(v2)
        v4 = self.conv1_bn(v3)
        v5 = self.conv2(v4)
        v6 = self.conv2_bn(v5)
        v7 = self.conv3(v6)
        v8 = self.conv3_bn(v7)
        v9 = self.conv4(v8)
        v10 = self.conv4_bn(v9)
        v11 = self.tanh(v10)
        return v11
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
