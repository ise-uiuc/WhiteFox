
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=2, groups=2)
        self.conv2 = torch.nn.Conv2d(64, 128, 1, stride=1, bias=False)
        self.conv3 = torch.nn.Conv2d(128, 16, 3, padding=1, dilation=1, groups=2, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3.sigmoid()
        v5 = v3 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
