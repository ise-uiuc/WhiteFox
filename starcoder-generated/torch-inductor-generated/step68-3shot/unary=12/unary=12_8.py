
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(32, 16, 3, stride=1, padding=2, dilation=1)
        self.conv_3 = torch.nn.Conv2d(16, 8, 3, stride=1, padding=1, dilation=2)
        self.conv_4 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=2, dilation=1)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = F.sigmoid(v1)
        v3 = self.conv_2(v2)
        v4 = F.sigmoid(v3)
        v5 = self.conv_3(v4)
        v6 = F.sigmoid(v5)
        v7 = self.conv_4(v6)
        v8 = F.sigmoid(v7)
        v9 = v8 * v1
        return v9
# Inputs to the model
x2 = torch.randn(1, 1, 64, 64)
