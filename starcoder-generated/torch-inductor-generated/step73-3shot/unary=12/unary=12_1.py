
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(192, 640, (3, 3), stride=(1, 1), padding=(1, 1), dilation=1, groups=1)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.conv2 = torch.nn.Conv2d(640, 320, (7, 7), stride=(1, 1), padding=(3, 3), dilation=3, groups=1)
        self.sigmoid2 = torch.nn.Sigmoid()
        self.conv3 = torch.nn.Conv2d(320, 160, (7, 7), stride=(1, 1), padding=(3, 3), dilation=3, groups=1)
        self.sigmoid3 = torch.nn.Sigmoid()
        self.conv4 = torch.nn.Conv2d(160, 3, (1, 1), stride=(1, 1), padding=(0, 0), dilation=1, groups=1)
        self.pool5 = torch.nn.AdaptiveAvgPool2d(1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.sigmoid1(v1)
        v3 = v1 * v2
        v4 = self.conv2(v3)
        v5 = self.sigmoid2(v4)
        v6 = v4 * v5
        v7 = self.conv3(v6)
        v8 = self.sigmoid3(v7)
        v9 = v7 * v8
        v10 = self.conv4(v9)
        v11 = self.pool5(v10)
        v12 = torch.flatten(v11, 1)
        return v12
# Inputs to the model
x1 = torch.randn(1, 192, 64, 64)
