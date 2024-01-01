
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(64)
    def forward(self, x1):
        v1 = self.bn1(self.conv1(x1))
        v2 = self.conv2(v1)
        v3 = self.bn2(v2)
        v4 = torch.nn.functional.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
