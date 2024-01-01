
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(512, 256, 7, stride=4, padding=2)
        self.conv3 = torch.nn.Conv2d(256, 256, 7, stride=3, padding=1)
        self.conv4 = torch.nn.Conv2d(256, 256, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = self.conv3(v7)
        v9 = self.conv4(v8)
        v10 = v9 * 0.31291076608655364
        v11 = v9 * 0.6352679536523097
        v12 = torch.erf(v11)
        v13 = v12 + 1
        v14 = v10 * v13
        return v14
# Inputs to the model
x1 = torch.randn(1, 512, 64, 64)
