
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 32, 3, stride=2, padding=1) #1, 32, 32, 32
        self.conv2 = torch.nn.Conv2d(32, 16, 5, padding=2, dilation=2) #1, 16, 62, 62
        self.conv3 = torch.nn.Conv2d(16, 8, 7, padding=3, dilation=3) #1, 8, 112, 112
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = torch.sigmoid(v3)
        v5 = v3 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 64, 256, 256)
