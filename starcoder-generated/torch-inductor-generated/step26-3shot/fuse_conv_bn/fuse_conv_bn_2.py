
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_1 = torch.nn.Conv2d(3, 64, 3, bias=False)
        self.batchnorm1d_1 = torch.nn.BatchNorm2d(64)
        self.conv2d_2 = torch.nn.Conv2d(64, 64, 3)
        self.batchnorm1d_2 = torch.nn.BatchNorm1d(64)
        self.conv2d_3 = torch.nn.Conv2d(64, 32, 2)
        self.batchnorm1d_3 = torch.nn.BatchNorm1d(32)
        self.conv2d_4 = torch.nn.Conv2d(288, 64, 3, padding=1)
        self.batchnorm1d_4 = torch.nn.BatchNorm1d(64)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.batchnorm1d_1(x)
        x = self.conv2d_2(x)
        x = self.batchnorm1d_2(x)
        x = torch.add(x, 0.1)
        x = self.conv2d_3(x)
        x = self.batchnorm1d_3(x)
        x = self.conv2d_4(x)
        x = self.batchnorm1d_4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
