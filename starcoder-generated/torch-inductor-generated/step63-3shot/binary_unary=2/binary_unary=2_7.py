
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool  = torch.nn.MaxPool2d(2, stride=1, padding=0)
        self.conv = torch.nn.Conv2d(1, 32, 2, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 64, 2, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 32, 2, stride=2, padding=0)
        self.conv4 = torch.nn.Conv2d(32, 128, 2, stride=2, padding=0)
        self.conv5 = torch.nn.Conv2d(128, 128, 2, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.pool(x1)
        v2 = self.conv(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        v5 = self.conv4(v4)
        v6 = self.conv5(v5)
        v7 = v6 - 0.5
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 4, 5)
