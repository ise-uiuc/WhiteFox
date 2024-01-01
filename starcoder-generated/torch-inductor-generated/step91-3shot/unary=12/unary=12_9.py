
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(224, 24, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(24, 24, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(24, 24, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(24, 24, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(24, 24, 1, stride=1, padding=0)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        v5 = self.conv4(v4)
        v6 = self.sigmoid(v5)
        v7 = v5 * v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 224, 64, 64)
