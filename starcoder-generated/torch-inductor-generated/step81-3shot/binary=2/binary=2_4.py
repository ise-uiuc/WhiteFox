
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 1, stride=1, padding=0)
        self.bn = torch.nn.BatchNorm2d(16)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn(x1)
        v3 = self.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 15, 5)
