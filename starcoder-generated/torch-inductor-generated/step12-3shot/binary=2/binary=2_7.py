
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3x3_1 = torch.nn.Conv2d(3, 64, 3, stride=2)
        self.conv1x3_1 = torch.nn.Conv2d(3, 24, 1, stride=1)
        self.conv3x3_2 = torch.nn.Conv2d(24, 64, 3, stride=1)
        self.conv1x3_2 = torch.nn.Conv2d(64, 24, 1, stride=1)
        self.conv3x3_3 = torch.nn.Conv2d(24, 64, 3, stride=2)
        self.conv1x3_3 = torch.nn.Conv2d(64, 24, 1, stride=1)
        self.conv3x3_4 = torch.nn.Conv2d(24, 64, 3, stride=1)
        self.conv1x3_4 = torch.nn.Conv2d(64, 24, 1, stride=1)
    def forward(self, x, x1):
        v1 = self.conv3x3_1(x)
        v2 = self.conv1x3_1(x1)
        out = x + x1
        v1 = self.conv3x3_3(v2) + (out*3)
        v2 = self.conv3x3_4(v1) - (out*4)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 8, 8)
x1 = torch.randn(1, 3, 8, 8)
