
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 22, 3, stride=1)
        self.conv2 = torch.nn.Conv2d(22, 5, 3, stride=2)
        self.conv3 = torch.nn.Conv2d(5, 7, 5, stride=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.nn.functional.max_pool2d(v1, 2, padding=1, ceil_mode=False)
        v3 = v2 * 0.5
        v4 = v2 * 0.7071067811865476
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        v8 = self.conv2(v7)
        v9 = torch.nn.functional.interpolate(v8, scale_factor=2, mode='nearest', align_corners=None)
        v10 = torch.nn.functional.interpolate(v7, scale_factor=2, mode='nearest', align_corners=None)
        v11 = v10 * 0.5
        v12 = v10 * 0.7071067811865476
        v13 = torch.erf(v12)
        v14 = v13 + 1
        v15 = v11 * v14
        v16 = self.conv3(v15)
        return v16
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
