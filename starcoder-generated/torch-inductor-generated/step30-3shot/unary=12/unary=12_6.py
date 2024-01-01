
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1, dilation=1, bias=False)
        self.conv_2 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0, dilation=1, bias=False)
        self.conv_3 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1, dilation=1, bias=False)
        self.conv_4 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0, dilation=1, bias=False)
        self.conv_5 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1, dilation=1, bias=False)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.conv_2(v1)
        v3 = self.conv_3(v2)
        v4 = self.conv_4(v3)
        v5 = self.conv_5(v4)
        v6 = v5.sigmoid()
        v7 = v5 * v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
