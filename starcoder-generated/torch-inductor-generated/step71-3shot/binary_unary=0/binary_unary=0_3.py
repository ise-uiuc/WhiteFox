
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(1, 8, 2, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(8, 16, 4, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(16, 32, 2, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 4, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        v5 = self.conv4(v4)
        v6 = torch.cat([v5, v5, v5], axis=0)
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
