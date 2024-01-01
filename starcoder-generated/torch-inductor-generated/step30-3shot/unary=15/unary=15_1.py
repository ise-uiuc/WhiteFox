
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(3, stride=2, padding=1)
        self.conv1 = torch.nn.Conv2d(3, 2, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(2)
    def forward(self, x1):
        v1 = self.pool(x1)
        v2 = self.bn1(self.conv1(v1))
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
