
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)
        self.bn = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = self.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
