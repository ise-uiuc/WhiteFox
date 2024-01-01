
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, (3, 3), stride=2, bias=False)
        self.bn = torch.nn.BatchNorm2d(8)
        self.conv2 = torch.nn.Conv2d(8, 8, (3, 3), stride=1, bias=False)
        self.relu = torch.nn.ReLU()
        self.pool3d = torch.nn.MaxPool2d((3, 3), stride=2)
        self.conv3 = torch.nn.Conv2d(8, 8, (3, 3), stride=1, bias=False)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.pool3d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        y = self.avg_pool(x)
        return y
# Inputs to the model
x = torch.randn(1, 1, 16, 16)
