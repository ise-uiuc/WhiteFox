
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 1, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(4, 4, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(4)
    def forward(self, x1):
        v1 = torch.relu(self.bn1(self.conv1(x1)))
        v2 = torch.relu(self.bn1(self.conv2(v1)))
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
