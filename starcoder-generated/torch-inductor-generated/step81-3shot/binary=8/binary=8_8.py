
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(8, 16, 1, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(8, 16, 1, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(16, 64, 1, stride=1, padding=1)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = self.conv3(v1 + v2)
        v4 = self.conv4(v1 + v3)
        v5 = self.conv5(x3)
        v6 = self.conv6(v4 + v5)
        v7 = self.conv7(v6 + v5)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
x3 = torch.randn(1, 3, 32, 32)
