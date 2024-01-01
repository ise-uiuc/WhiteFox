
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(17, 17, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(17, 96, 15, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(96, 384, 6, stride=3, padding=1)
        self.conv4 = torch.nn.Conv2d(384, 896, 128, stride=1, padding=124)
        self.conv5 = torch.nn.Conv2d(896, 896, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
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
x1 = torch.randn(1, 17, 61, 61)
