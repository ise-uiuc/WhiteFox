
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 12, 3, stride=1, padding=1, dilation=1)
        self.conv_2 = torch.nn.Conv2d(12, 18, 3, stride=2, padding=0, dilation=1)
        self.conv_3 = torch.nn.Conv2d(18, 18, 3, stride=4, padding=8, dilation=1)
        self.conv_4 = torch.nn.Conv2d(18, 18, 3, stride=8, padding=16, dilation=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.add = torch.add
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.conv_2(v1)
        v3 = self.conv_3(v2)
        v4 = self.conv_4(v3)
        v5 = self.sigmoid(v4)
        v6 = self.add(v4, v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
