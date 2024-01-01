
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.avgpool = torch.nn.AvgPool2d(3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.conv_2(v1)
        v3 = self.avgpool(v2)
        v4 = 3 + v3
        v5 = torch.clamp_min(v4, 0)
        v6 = torch.clamp_max(v5, 6)
        v7 = v3 * v6
        v8 = v7 / 6
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
