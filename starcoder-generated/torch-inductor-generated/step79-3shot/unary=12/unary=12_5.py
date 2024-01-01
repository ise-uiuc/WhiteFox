
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 10, 3, stride=1, padding=1, dilation=1)
        self.conv_0 = torch.nn.Conv2d(10, 10, 3, stride=1, padding=2, dilation=2)
        self.conv_1 = torch.nn.Conv2d(10, 10, 3, stride=1, padding=4, dilation=4)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_0(x1)
        v3 = self.conv_1(x1)
        v4 = v1.sigmoid()
        v5 = v2.sigmoid()
        v6 = v3.sigmoid()
        v7 = v1 * v4
        v8 = v2 * v5
        v9 = v3 * v6
        return v7 + v8 + v9
# Inputs to the model
x1 = torch.randn(1, 10, 240, 320)
