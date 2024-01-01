
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(2)
        self.conv1 = torch.nn.Conv2d(2, 4, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(2, 4, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(self.bn(x1))
        v2 = torch.sigmoid(v1)
        v3 = self.bn(self.conv2(x1))
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 128, 128)
