
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 7, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(7, 4, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(12, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = torch.nn.functional.interpolate(v6, scale_factor=[1, 1.1111], mode='nearest', align_corners=False)
        v8 = self.conv2(v7)
        v9 = v8 * 0.5
        v10 = v8 * 0.7071067811865476
        v11 = torch.erf(v10)
        v12 = v11 + 1
        v13 = v9 * v12
        v14 = self.conv3(torch.cat((v13, v6), 1))
        return v14
# Inputs to the model
x1 = torch.randn(1, 1, 14, 7)
