
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.conv = torch.nn.Conv2d(8, 16, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.bn1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv(v2)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(4, 8, 57, 57)
