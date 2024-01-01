
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(137, 154, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(154, 83, 5, stride=6, padding=1, groups=83)
        self.conv3 = torch.nn.Conv2d(83, 121, 6, stride=4, padding=3, groups=83, dilation=3)
        self.conv4 = torch.nn.Conv2d(121, 245, 4, stride=2, padding=1, groups=121, dilation=1)
        self.conv5 = torch.nn.Conv2d(245, 140, 1, stride=1)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = torch.tanh(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(x4)
        x6 = torch.tanh(x5)
        return self.conv5(x6)
# Inputs to the model
x = torch.randn(1, 137, 56, 56)
