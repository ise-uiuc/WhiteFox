
# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 64, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(8, 64, 3, stride=2, padding=0)
        self.conv4 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=0)
        self.conv5 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=0)
        self.conv5 = torch.nn.Conv2d(64, 32, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(x1)
        v3 = self.conv2(v1)
        v4 = self.conv3(v1)
        v5 = self.conv4(v3)
        v6 = self.conv5(v5)
        v7 = v2 + v6
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
