
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, 5, stride=1, padding=1, groups=1, bias=False, dilation=1)
        self.max = max
        self.conv2 = torch.nn.Conv2d(16, 32, 13, stride=1, padding=5, groups=1, bias=True, dilation=1)
        self.relu = torch.nn.ReLU(inplace=False)
        self.min = min
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.clamp_max(v1, self.max)
        v3 = self.conv2(v2)
        v4 = self.relu(v3)
        v5 = torch.clamp_min(v4, self.min)
        return v5
min = 1.7763568394002505e-15
max = 1.0921236179323194e-17
# Inputs to the model
x1 = torch.randn(1, 1, 48, 48)
