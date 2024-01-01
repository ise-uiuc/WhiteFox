
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)
        self.conv3 = torch.nn.Conv2d(1, 1, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(1, 1, 1)
        self.conv5 = torch.nn.Conv2d(1, 1, 3, stride=2, padding=1)
        self.conv6 = torch.nn.Conv2d(1, 1, 1)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x2)
        x5 = self.conv5(x3)
        x6 = self.conv6(x5)
        return x6
# Inputs to the model
x = torch.randn(1, 1, 3, 3)
x1 = torch.randn(1, 1, 28, 28)
