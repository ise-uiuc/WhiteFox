
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 2, 3)
        self.bn1 = torch.nn.BatchNorm2d(2)
        self.relu = torch.nn.ReLU6(inplace=True)
        self.relu1 = torch.nn.ReLU6(inplace=True)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = self.relu(v2)
        v4 = v3.permute(0, 3, 2, 1)
        v5 = self.relu1(v4)
        v6 = torch.randn((v5.size()[0], 2, 3, 2)) + v5
        v7 = torch.randn((v5.size()[0], 2, 3, 2)) + v5
        v8 = v6 + v7
        v9 = v6 - v7
        v10 = self.avgpool(v8)
        return v1.permute(0, 3, 1, 2)
# Inputs to the model
x1 = torch.randn(1, 2, 12, 12, device='cpu')
