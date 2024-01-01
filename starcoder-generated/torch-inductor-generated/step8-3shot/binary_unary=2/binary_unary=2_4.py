
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 16, 1, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(16, 32, 1, stride=1, padding=4)
        self.conv4 = torch.nn.Conv2d(64, 20, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = torch.add(v1, v3)
        v5 = torch.sub(v2, v3)
        v6 = torch.cat([v4, v5], dim=1)
        v7 = self.conv4(v6)
        v8 = v5 + v7
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
