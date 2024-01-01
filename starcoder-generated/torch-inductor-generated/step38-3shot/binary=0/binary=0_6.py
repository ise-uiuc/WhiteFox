
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(32, 16, stride=2, kernel_size=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        return v2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 32, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(32, 24, kernel_size=1)
        self.conv3 = torch.nn.Conv2d(24, 4, kernel_size=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
