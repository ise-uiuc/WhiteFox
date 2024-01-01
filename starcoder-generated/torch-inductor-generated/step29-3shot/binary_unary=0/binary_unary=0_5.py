
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv_1 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv_2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
    def forward(self, x1, x3):
        v1 = self.conv(x1)
        v2 = v1 + x3
        v3 = torch.relu(v2)
        v4 = self.conv_1(v3)
        v5 = v4 + x3
        v6 = torch.relu(v5)
        v7 = self.conv_2(v6)
        v8 = v7 + v7
        v9 = torch.relu(v8)
        return v9
