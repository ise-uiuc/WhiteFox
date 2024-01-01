
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 2, stride=1, padding=1, dilation=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 2, stride=2)
        self.conv4 = torch.nn.Conv2d(64, 128, 1, stride=1)
        self.conv5 = torch.nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(256, 512, 1, stride=1)
        self.conv7 = torch.nn.Conv2d(512, 16, 1, stride=1)
    def forward(self, x):
        y = self.conv1(x)
        y = torch.tanh(y)
        y = self.conv2(y)
        y = torch.tanh(y)
        y = self.conv3(y)
        y = torch.tanh(y)
        y = self.conv4(y)
        y = torch.tanh(y)
        y = self.conv5(y)
        y = torch.tanh(y)
        y = self.conv6(y)
        y = torch.tanh(y)
        y = self.conv7(y)
        y = torch.tanh(y)
        return y
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
