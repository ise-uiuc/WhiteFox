
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.clamp_min(v1, self.min)
        v3 = self.conv2(v2)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = -0.4
max = 0.6
# Inputs to the model
x = torch.randn(1, 3, 50, 50)
