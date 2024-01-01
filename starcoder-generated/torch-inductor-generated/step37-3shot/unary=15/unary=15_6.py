
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.bn1(self.conv1(x1))
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
