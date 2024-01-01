
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, 1, stride=1, padding=0)
        self.other_conv = torch.nn.Conv2d(10, 10, 3, stride=1, padding=0)
        self.pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.add_(3)
        v3 = v2.clamp(min=0)
        v4 = v3.clamp(max=6)
        v5 = v4/6
        v6 = self.other_conv(v5)
        v7 = v6.add_(3)
        v8 = v7.clamp(min=0)
        v9 = v8.clamp(max=6)
        v10 = v9/6
        v11 = self.pooling(v10)
        v12 = torch.flatten(v11, 1)
        return v12
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
