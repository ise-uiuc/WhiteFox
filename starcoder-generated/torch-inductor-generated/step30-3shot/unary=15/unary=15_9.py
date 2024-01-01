
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(3, 3, 5, stride=1, padding=2)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.bn2 = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        v1 = self.bn2(self.conv2(x1))
        v2 = self.bn1(self.conv1(v1))
        v3 = torch.relu(v2)
        v4 = torch.relu(v3)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
