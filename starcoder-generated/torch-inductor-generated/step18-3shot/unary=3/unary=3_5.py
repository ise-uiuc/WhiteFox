
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(256, 256, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(256, int(256 / 2), 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(int(256 / 2), 256, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.7071067811865476
        v3 = torch.erf(v2)
        v4 = v3 + 1
        v5 = self.conv2(v4)
        v6 = v5 * 0.5
        v7 = v5 * 0.7071067811865476
        v8 = torch.erf(v7)
        v9 = v8 + 1
        v10 = self.conv3(v9)
        v11 = v5 * 0.5
        v12 = v5 * 0.7071067811865476
        v13 = torch.erf(v12)
        v14 = v13 + 1
        v15 = self.conv4(v14)
        return v15
# Inputs to the model
x1 = torch.randn(16, 256, 16, 16)
