
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(96, 96, 3, stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv5 = torch.nn.Conv3d(128, 4, 3, stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv6 = torch.nn.Conv2d(3, 64, 3, 1, 1)
        self.conv7 = torch.nn.Conv2d(64, 32, 3, 1, 1)
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm2d(32)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv6(v1)
        v3 = self.relu(v2)
        v4 = self.conv7(v3)
        v5 = self.bn(v4)
        v6 = self.conv5(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 96, 32, 32, 32)
