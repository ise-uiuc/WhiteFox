
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = self.relu(v2)
        v4 = v3.max_pool2d(3, stride=2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 512, 512)
