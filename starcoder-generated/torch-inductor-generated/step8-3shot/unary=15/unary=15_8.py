
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(16)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = torch.relu(v2)
        v4 = torch.nn.functional.interpolate(v3, size=(512, 512), mode='bilinear', align_corners=False)
        v5 = torch.nn.functional.interpolate(v3, scale_factor=2.0, mode='nearest')
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 512, 512)
