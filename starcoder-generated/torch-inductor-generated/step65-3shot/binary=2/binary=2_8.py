
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 16, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 112, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(112, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 - 10
        return v4
# Inputs to the model
x = torch.randn(1, 8, 256, 256)
