
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1a = torch.nn.Conv2d(3, 10, 4, stride=1, padding=0)
        self.bn1a = torch.nn.BatchNorm2d(10)
        self.conv1b = torch.nn.Conv2d(10, 10, 4, stride=1, padding=0)
        self.bn1b = torch.nn.BatchNorm2d(10)
    def forward(self, x1):
        v1 = self.bn1a(self.conv1a(x1))
        v2 = self.bn1b(self.conv1b(v1))
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
