
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 3, 7, stride=2, padding=3, groups=1)
        self.conv2 = torch.nn.Conv2d(3, 1, 7, stride=2, padding=3, groups=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.nn.functional.interpolate(v1, size=[64, 64], mode='bilinear', align_corners=False)
        v3 = self.conv2(v2)
        v4 = torch.nn.functional.interpolate(v3, size=[64, 64], mode='bilinear', align_corners=False)
        return v4
# Inputs to the model
x = torch.randn(1, 10, 64, 64)
