
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(40, 64, 5, stride=1, padding=2)
        self.avgpool = torch.nn.AvgPool2d(7, stride=1)
        self.conv_1 = torch.nn.Conv2d(64, 128, 1, stride=1)
        self.conv_2 = torch.nn.Conv2d(128, 256, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.avgpool(v1)
        v3 = self.conv_1(v2)
        v4 = self.conv_2(v3)
        v5 = torch.add(v1, v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 40, 45, 64)
