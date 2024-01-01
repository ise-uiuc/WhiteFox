
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
    def forward(self, x2, x3):
        v5 = self.conv1(x2)
        v7 = self.bn1(v5)
        v8 = torch.sigmoid(v7)
        v9 = v8 + x3
        return v9
# Inputs to the model
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 8, 64, 64)
