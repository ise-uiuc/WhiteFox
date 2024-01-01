
class Model(torch.nn.Module):
    def __init__(self, min, max, conv1_stride=1, conv2_stride=3):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, 3, stride=conv1_stride, padding=1)
        self.conv2 = torch.nn.Conv2d(5, 8, 5, stride=conv2_stride, padding=0)
        self.min = min
        self.max = max
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 0.7
max = -0.6
# Inputs to the model
x = torch.randn(1, 1, 16, 16)
