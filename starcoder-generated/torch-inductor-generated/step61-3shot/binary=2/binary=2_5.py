
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_0 = torch.nn.Conv2d(24, 16, 1, stride=1, padding=1)
        self.conv_1 = torch.nn.Conv2d(24, 24, 1, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(24, 24, 1, stride=1, padding=1)
        self.conv_3 = torch.nn.Conv2d(24, 24, 1, stride=1, padding=1)
        self.conv_4 = torch.nn.Conv2d(24, 24, 1, stride=1, padding=1)
        self.conv_5 = torch.nn.Conv2d(24, 24, 1, stride=1, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        v0 = self.conv_0(x)
        v1 = self.conv_1(x)
        v2 = self.conv_2(x)
        v3 = self.conv_3(x)
        v4 = self.conv_4(x)
        v5 = self.conv_5(x)
        v6 = self.sigmoid(v0)
        v7 = v0 - v1
        v8 = v0 - v2
        v9 = v0 - v3
        v10 = v0 - v4
        v11 = v0 - v5
        v12 = v0 - v6
        return v6, v7, v8, v9, v10, v11, v12
# Inputs to the model
x = torch.randn(2, 24, 1, 56)
