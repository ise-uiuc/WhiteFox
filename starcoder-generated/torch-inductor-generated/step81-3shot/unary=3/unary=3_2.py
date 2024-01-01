
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(83, 37, 3, stride=1, padding=0, dilation=1, groups=7)
        self.conv2 = torch.nn.Conv2d(37, 85, 5, stride=4, padding=3, dilation=1, groups=2)
        self.conv3 = torch.nn.ConvTranspose2d(85, 128, 2, stride=2, padding=1, dilation=1, groups=4)
        self.conv4 = torch.nn.Conv3d(128, 90, 3, stride=4, padding=1, dilation=5, groups=8)
        self.conv5 = torch.nn.Conv1d(90, 46, 1, stride=3, padding=2, dilation=3, groups=4)
        self.relu1 = torch.nn.ReLU(inplace=False)
    def forward(self, x1):
        v4 = torch.erf(x1)
        v2 = self.relu1(self.conv(v4))
        v3 = v2 * 0.5
        v1 = torch.erf(self.conv2(v3))
        v5 = self.conv3(v1)
        v6 = self.relu1(self.conv4(v5))
        v7 = self.conv5(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 83, 111)
