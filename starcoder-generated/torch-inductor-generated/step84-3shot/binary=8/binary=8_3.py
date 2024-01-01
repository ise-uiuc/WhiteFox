
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = v1 + x2
        v3 = self.bn1(v2)
        v4 = v3 + x1
        x3 = torch.tensor([1])
        v10 = torch.cat([v4, v4, v3, x3, v1, v3], 1)
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
