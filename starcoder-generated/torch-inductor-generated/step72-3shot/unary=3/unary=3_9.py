
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        y = self.conv(x1)
        y = self.conv2(y)
        y = self.conv3(y)
        v1 = y * 0.5
        v2 = y * 0.7071067811865476
        v3 = torch.erf(v2)
        v4 = v3 + 1
        v5 = v1 * v4
        v6 = self.conv4(v5)
        v7 = v6 * 0.5
        v8 = v6 * 0.7071067811865476
        v9 = torch.erf(v8)
        v10 = v9 + 1
        v11 = v7 * v10
        v12 = self.conv5(v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 5, 32, 32)
