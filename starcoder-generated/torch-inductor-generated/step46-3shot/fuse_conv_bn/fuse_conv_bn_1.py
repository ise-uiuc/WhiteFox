
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, (3, 3), stride=2, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(8, 8, (3, 3), stride=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(8, 8, (3, 3), stride=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(8)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        y = self.avg_pool(x)
        return y
# Inputs to the model
x = torch.randn(1, 1, 16, 16)
