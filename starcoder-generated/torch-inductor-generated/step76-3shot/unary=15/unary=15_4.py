
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(144, 24, 1, stride=1, padding=0)
        self.conv0 = torch.nn.Conv2d(3, 192, 7, stride=2, padding=2)
        self.conv1 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = x1.permute(0, 2, 3, 1)
        v4 = self.conv0(v3)
        v5 = torch.relu(v4)
        v6 = v2.permute(0, 2, 3, 1)
        v7 = torch.cat([v6, v5], 3)
        v8 = v7 + 1
        v9 = v8.permute(0, 3, 1, 2)
        v10 = self.conv1(v9)
        v11 = torch.relu(v10)
        return v11
# Inputs to the model
x1 = torch.randn(17, 3, 50, 50)
