
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 20, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(6, 3, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(20, 6, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(20, 1, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(14, 10, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        v8 = self.conv3(v7)
        v9 = self.conv4(v8)
        v10 = v8 * 0.5
        v11 = v8 * 0.7071067811865476
        v12 = torch.erf(v11)
        v13 = v12 + 1
        v14 = v10 * v13
        v15 = self.conv5(v14)
        return v15
# Inputs to the model
x1 = torch.randn(1, 3, 72, 72)
