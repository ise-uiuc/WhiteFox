
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, dilation=2)
        self.conv2 = nn.Conv2d(32, 32, 3, dilation=2)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, dilation=8)
        self.conv4 = nn.Conv2d(31, 32, 3, dilation=2)
    def forward(self, x):
        y = torch.nn.functional.avg_pool2d(self.conv1(x), 2)
        y = torch.nn.functional.relu(self.conv2(y))
        y = torch.nn.functional.max_pool2d(self.conv3(y), 2)
        y1 = torch.nn.functional.avg_pool2d(self.conv4(y), 2)
        y1 = self.conv3(y1)
        y2 = self.conv2(torch.tanh(y1))
        y3 = torch.tanh(y1) - y2
    return torch.cat([y, y1, y2, y3], 1)
# Inputs to the model
x = torch.randn(1, 1, 224, 224)
