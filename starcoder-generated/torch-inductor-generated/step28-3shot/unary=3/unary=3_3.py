
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(16, 64, 5, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv5 = torch.nn.Conv2d(64, 64, 5, stride=1, padding=2)
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
        v10 = self.conv5(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
