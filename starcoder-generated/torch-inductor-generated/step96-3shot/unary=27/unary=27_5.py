
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(2, 3, 1, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = torch.clamp_min(v3, self.min)
        v5 = torch.clamp_max(v4, self.max)
        return v5
min = 4.0
max = -1.0
# Inputs to the model
x = torch.randn(1, 2, 10, 7)
