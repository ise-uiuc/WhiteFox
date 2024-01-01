
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.conv3 = torch.nn.Conv2d(8, 16, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.bn1(v2)
        v4 = self.conv3(v3)
        return F.relu(v4)
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
