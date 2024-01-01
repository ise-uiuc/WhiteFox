
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, 3, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(1, 2, 2, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = self.conv2(v2)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 0.4
max = 0.43
# Inputs to the model
x1 = torch.randn(1, 1, 10, 10)
