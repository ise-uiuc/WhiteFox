
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(16)
    def forward(self, x):
        v = []
        v.append(self.conv1(x))
        v.append(self.conv2(x))
        v.append(self.conv1(x))
        v.append(self.conv2(x))
        v1 = torch.cat(v, dim=1)
        return self.bn(v1)
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
