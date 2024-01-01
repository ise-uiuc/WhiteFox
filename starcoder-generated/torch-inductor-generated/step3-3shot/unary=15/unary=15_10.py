
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 6, 3, padding=1)
        self.bn = torch.nn.BatchNorm2d(6)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        y = torch.relu(v2)
        return y
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
