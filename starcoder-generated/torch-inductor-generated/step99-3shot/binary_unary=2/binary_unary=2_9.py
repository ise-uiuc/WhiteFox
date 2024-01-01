
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(32)

        self.conv2 = torch.nn.Conv2d(32, 64, 5, stride=2, padding=2, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(64)

        self.conv3 = torch.nn.Conv2d(64, 64, 5, stride=2, padding=2, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(64)

        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.linear1 = torch.nn.Linear(64, 10, bias=True)
    def forward(self, x1):
        v1 = F.avg_pool2d(self.bn1(self.conv1(x1)), 2)
        v2 = F.avg_pool2d(self.bn2(self.conv2(v1)), 2)
        v3 = F.avg_pool2d(self.bn3(self.conv3(v2)), 2)
        v4 = self.bn3(self.gap(v3)).flatten(1)
        v5 = F.relu(self.linear1(v4))
        x2 = v5 - 10
        return x2
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
