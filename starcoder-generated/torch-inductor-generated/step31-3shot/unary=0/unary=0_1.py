
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=0, dilation=1, groups=3)
        self.conv2 = torch.nn.Conv2d(32, 2, 5, stride=1, padding=1, dilation=2, groups=3)
        self.conv3 = torch.nn.Conv2d(6, 8, 5, stride=1, padding=0, dilation=1, groups=3)
    def forward(self, x3):
        v1 = self.conv1(x3)
        v2 = self.conv2(v1)
        v3 = self.conv3(x3)
        v4 = torch.cat((v2, v3), dim=1)
        return v4
# Inputs to the model
x3 = torch.randn(2, 6, 128, 128)
