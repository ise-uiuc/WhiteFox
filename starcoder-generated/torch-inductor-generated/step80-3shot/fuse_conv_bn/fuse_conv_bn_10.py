
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.bn2 = torch.nn.BatchNorm2d(3)
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.relu = torch.nn.ReLU(inplace=True)
        self.max_pool2d = torch.nn.MaxPool2d(2)
        self.bn3 = torch.nn.BatchNorm2d(3, affine=False)
        self.avg_pool_3d = torch.nn.AvgPool3d(2)
    def forward(self, x1):
        s = self.bn1(x1)
        t = self.bn2(s)
        y = self.conv(t)
        z = self.relu(y)
        a = self.max_pool2d(z)
        b = self.bn3(a)
        c = self.avg_pool_3d(b)
        return c
# Inputs to the model
x1 = torch.randn(1, 3, 6, 6)
