
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 2, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 1, 2, stride=2, padding=1)
        self.min = min
        self.max = max
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = -0.7
max = 0.6
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
