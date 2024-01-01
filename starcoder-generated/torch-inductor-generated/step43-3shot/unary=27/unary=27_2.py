
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 2, 3, stride=4, padding=3)
        self.conv2 = torch.nn.Conv2d(2, 1, 3, stride=4, padding=3)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = self.conv2(v3)
        return v4
min = -3
max = 0
# Inputs to the model
x1 = torch.randn(1, 4, 28, 28)
