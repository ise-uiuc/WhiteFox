
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 1, stride=1)
        self.bn = torch.nn.BatchNorm2d(6)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn(v1)
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 6, 6)
