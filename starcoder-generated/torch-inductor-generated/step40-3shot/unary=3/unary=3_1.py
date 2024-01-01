
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 3, 5, stride=2, padding=2)
        self.conv4 = torch.nn.Conv2d(3, 3, 7, stride=2, padding=3)
        self.conv5 = torch.nn.Conv2d(3, 3, 7, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = self.conv3(v7)
        v9 = v8 * 0.5
        v10 = v8 * 0.7071067811865476
        v11 = torch.erf(v10)
        v12 = v11 + 1
        v13 = v9 * v12
        v14 = self.conv4(v13)
        v15 = self.conv5(v14)
        return v15
# Inputs to the model
x1 = torch.randn(1, 512, 14, 14)
