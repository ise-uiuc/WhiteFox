
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.bn2 = torch.nn.BatchNorm2d(32)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        out = torch.relu(v2)
        out = self.conv2(out)
        out = torch.relu(out)
        out = self.conv3(out)
        out = self.bn2(out)
        out = torch.relu(out)
        return out
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
