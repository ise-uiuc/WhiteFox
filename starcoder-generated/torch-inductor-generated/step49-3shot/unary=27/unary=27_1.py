
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 7, stride=2, padding=3, groups=1, bias=True, dilation=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.clamp_min(v3, self.min)
        v5 = torch.clamp_max(v4, self.max)
        return v5
min = -3.552713678800501e-15
max = 1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
