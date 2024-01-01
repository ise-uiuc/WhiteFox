
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, stride=3, padding=0)
        self.conv1 = torch.nn.Conv2d(8, 16, 1, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 16, 3, stride=3, padding=4)
        self.conv3 = torch.nn.Conv2d(1, 3, 2, stride=4, padding=0)
    def forward(self, x):
        v1 = self.conv(x).mean()
        x1 = v1 + 5.0**1.5
        v2 = self.conv1(x1).mean()
        x2 = v2 + 0.0005
        v3 = x2 - 100.0
        v4 = self.conv2(v3).mean()
        x3 = v4 + 3.8
        v5 = self.conv3(x3).mean()
        x4 = v5 - 0.5
        v6 = x4 - -200.0**200.0
        return x4
# Inputs to the model
x = torch.randn(1, 4, 15, 15)
