
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.norm1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.norm2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.norm3 = torch.nn.BatchNorm2d(128)
        self.conv4 = torch.nn.Conv2d(128, 128, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.norm1(v1)
        v3 = self.conv2(v2)
        v4 = self.norm2(v3)
        v5 = self.conv3(v4)
        v6 = self.norm3(v5)
        v7 = self.conv4(v6)
        v8 = torch.sigmoid(v7)
        v9 = torch.mul(v7, v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
