
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, 3, stride=1, padding=1, dilation=2)
        self.conv2 = torch.nn.Conv2d(5, 10, 3, stride=1, padding=1, dilation=2)
        self.conv3 = torch.nn.Conv2d(10, 20, 3, stride=1, padding=1, dilation=2)
        self.conv4 = torch.nn.Conv2d(20, 40, 3, stride=1, padding=1, dilation=2)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = self.conv3(x3)
        x5 = self.conv4(x4)
        y = x5 - 0.01
        y = F.relu(y)
        return y
# Inputs to the model
x1 = torch.randn(1, 3, 24, 24)
