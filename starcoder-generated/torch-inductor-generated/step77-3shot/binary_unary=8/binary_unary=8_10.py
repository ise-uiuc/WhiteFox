
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=3, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 4, 3, stride=3, padding=0)
        self.bn = torch.nn.BatchNorm2d(8)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        v5 = self.bn(v4)
        out = torch.reshape(v5, -1)
        return out
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
