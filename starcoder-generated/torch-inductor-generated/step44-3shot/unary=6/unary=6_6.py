
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.batchnorm_1 = torch.nn.BatchNorm2d(3)
        self.conv_1 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.batchnorm_2 = torch.nn.BatchNorm2d(3)
        self.conv_2 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.batchnorm_3 = torch.nn.BatchNorm2d(3)
        self.conv_3 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.batchnorm_1(self.conv(x1)) + self.batchnorm_2(self.conv(x1)) + self.batchnorm_3(self.conv(x1))
        v2 = self.conv_1(self.conv(x1)) + self.conv_2(self.conv(x1)) + self.conv_3(self.conv(x1))
        v3 = v1 + v2
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        v6 = self.batchnorm_1(v5) + self.batchnorm_2(v5) + self.batchnorm_3(v5)
        v7 = self.conv_1(v5) + self.conv_2(v5) + self.conv_3(v5)
        v8 = v6 + v7
        v9 = v8 / 6
        v10 = self.conv(x1) + v9
        v11 = self.conv(x1) + v9
        return self.conv_1(v10)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
