
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(4, 5, 1, stride=1, padding=0)
        self.maxd = torch.nn.MaxPool2d(kernel_size=5, stride=2)
        self.maxi = torch.nn.MaxPool2d(kernel_size=7, stride=1)
        self.conv3 = torch.nn.Conv2d(5, 5, 1, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1) + v1
        v3 = self.maxd(v2)
        v4 = self.maxi(v1)
        v5 = torch.clamp_min(v3, self.min)
        v6 = torch.clamp_max(v5, self.max)
        v7 = torch.clamp_min(v6 + v4, self.min)
        v8 = self.conv3(v1)
        return v8
min = 0.1
max = 2.1
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
