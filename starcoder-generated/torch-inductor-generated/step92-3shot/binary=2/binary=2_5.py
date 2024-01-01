
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 32, 1, stride=1, padding=1)
        self.conv_1 = torch.nn.Conv2d(32, 64, 1, stride=1, padding=1)
        self.conv_3 = torch.nn.Conv2d(64, 128, 1, stride=1, padding=1)
        self.conv_4 = torch.nn.Conv2d(128, 16, 1, stride=1, padding=1)
        self.softmax = torch.nn.Softmax()
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1.view(6, 7744, 1)
        v3 = v2 - 1.0
        v4 = self.conv_1(v3.view(6, 1, 11, 11))
        v5 = v4.view(6, 16, 1, 1)
        v6 = v5 + 1.0
        v7 = self.conv_3(v6.view(6, 1, 5, 5))
        v8 = v7.view(6, 128, 1, 1)
        v9 = v8 - 1.0
        v10 = self.conv_4(v9)
        v11 = v10 + 1.0
        return self.softmax(v11)
# Inputs to the model
x = torch.randn(1, 8, 7, 7)
