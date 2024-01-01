
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1908, 1196, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(1196, 764, 7, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(764, 598, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(598, 764, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(764, 598, 7, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(598, 600, 1, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(600, 1600, 2, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = self.conv6(v5)
        v7 = self.conv7(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 1908, 119, 99)
