
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 33, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(3, 1, 31, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 1, 17, stride=3, padding=1)
        self.conv4 = torch.nn.Conv2d(3, 1, 31, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(3, 1, 27, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(3, 1, 3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(3, 1, 39, stride=1, padding=1)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.conv1(x1)
        x4 = (x2 + x3).add(x1)
        x5 = self.conv2(x1)
        x6 = self.conv3(x1)
        x7 = (x6 - x5).add(x4)
        x8 = self.conv4(x1)
        x9 = self.conv5(x1)
        x10 = (x8 - x9).add(x7)
        x11 = self.conv6(x1)
        x12 = self.conv7(x1)
        x13 = (x12 - x11).add(x10)
        return x13
# Inputs to the model
x1 = torch.randn(1, 3, 40, 20)
