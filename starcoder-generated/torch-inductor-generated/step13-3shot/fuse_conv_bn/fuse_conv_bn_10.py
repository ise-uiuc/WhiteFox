
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 2)
        self.bn = torch.nn.BatchNorm2d(1)
        self.avgpool = torch.nn.AvgPool2d(2)
        self.conv2 = torch.nn.Conv2d(1, 1, 2)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.avgpool(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
