
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(6, 12, 5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(12, 24, 5, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(24, 48, 5, stride=1, padding=2)
        self.conv5 = torch.nn.Conv2d(48, 96, 5, stride=1, padding=2)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(x4)
        x6 = self.conv5(x5)
        y = x6 - 249
        y = F.relu(y)
        return y
# Inputs to the model
x1 = torch.randn(1, 1, 24, 24)
