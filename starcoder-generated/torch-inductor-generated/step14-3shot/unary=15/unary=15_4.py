
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 2, stride=2, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.bn1(self.conv1(x1))
        v2 = self.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
