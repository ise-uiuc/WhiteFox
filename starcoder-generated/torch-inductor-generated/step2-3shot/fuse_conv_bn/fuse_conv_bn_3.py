
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 2)
        self.bn = torch.nn.BatchNorm2d(1)
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x):
        x1 = self.bn(x)
        x2 = self.linear(x1)
        return x2
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
# Model begins

# Model begins
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 3)
        self.conv1 = torch.nn.Conv2d(4, 7, 2, stride=1)
        self.conv2 = torch.nn.Conv2d(8, 6, 2, stride=2, bias=False)
        self.conv3 = torch.nn.Conv1d(2, 3, 1, bias=False)
        self.conv4 = torch.nn.Conv3d(3, 5, 3)
    def forward(self, x):
        x1 = self.conv4(self.conv3(self.conv2(self.conv1(self.conv(x)))))
        return x1
x = torch.randn(1, 4, 3)
