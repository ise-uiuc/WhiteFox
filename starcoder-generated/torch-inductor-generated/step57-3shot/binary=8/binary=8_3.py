
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=1, padding=1)
        self.gap = torch.nn.AdaptiveAvgPool2d((8, 8))
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = self.gap(v1)
        # 64-bit floating-point output; 4 dimensions.
        v4 = torch.nn.functional.cross_entropy(v3, torch.empty((0), dtype=torch.long))
        v5 = torch.nn.functional.interpolate(v2, size=(16, 16), mode='bilinear')
        v6 = v5 + v4
        return v6
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
