
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 64, 5, stride=3, padding=1)
        self.min = min
        self.max = max
    def forward(self, x):
        v1 = torch.clamp_min(x, self.min)
        v2 = self.conv2d(v1)
        v3 = torch.clamp_max(v2, self.max)
        s1 = torch.sum(v3)
        v4 = torch.clamp(s1, self.min, self.max)
        return v4
min = 1
max = 3
# Inputs to the model
x = torch.randn(6, 3, 28, 28)
