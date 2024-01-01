
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(16, 8, 1, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(8, 3, 3, stride=1, padding=1)
        self.conv8 = torch.nn.Conv2d(16, 3, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = self.conv3(x)
        v4 = self.conv4(x)
        v5 = self.conv5(v1)
        v6 = self.conv6(v3)
        v7 = self.conv7(v5)
        v8 = self.conv8(v6)
        return (v7 + v8, v7 - v8)
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
