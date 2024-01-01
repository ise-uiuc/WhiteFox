
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, (3, 3), stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = self.bn1(x2)
        x4 = self.relu(x3)
        x5 = self.conv(x4)
        x6 = self.relu(x5)
        x7 = self.bn1(x6)
        return x7
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
