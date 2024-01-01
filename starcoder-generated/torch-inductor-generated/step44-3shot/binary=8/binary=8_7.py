
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 16, 1, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 32, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(32, 64, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = self.conv6(v5)
        v7 = self.conv7(v6)
        v8 = v1.add(v6)
        v9 = v3 - v6
        v10 = torch.conv_transpose2d(v2, v7, padding=0, stride=1, groups=1)
        v11 = torch.cat([v5, v6, v7, v8, v9, v10], dim=1)
        return v11
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
