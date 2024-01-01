
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=2)
        self.conv5 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=2)
        self.conv6 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=2)
        self.conv7 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=2)
        self.conv8 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=2)
        self.conv9 = torch.nn.Conv2d(3, 8, 1, stride=2, padding=2)
    def forward(self, x1, x2, x3, x4, x5, x6):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x2)
        v4 = self.conv4(x2)
        v5 = self.conv5(x3)
        v6 = self.conv6(x3)
        v7 = self.conv7(x4)
        v8 = self.conv8(x4)
        v9 = self.conv9(x5)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 3, 64, 64)
x4 = torch.randn(1, 3, 64, 64)
x5 = torch.randn(1, 3, 64, 64)
x6 = torch.randn(1, 3, 64, 64)
