
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.bn1 = nn.BatchNorm1d(3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv2d(12, 6, 3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        y = self.conv2(x)
        y = self.bn2(y)
        z = self.conv3(y)
        z = z + x
        return z
# Inputs to the model
x = torch.randn(1, 1, 2, 2)
