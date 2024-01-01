
class Model(torch.nn.Module):
    def __init__(self, min, max, relu=True):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, 5, stride=2, padding=1)
        self.min = min
        self.max = max
        self.relu = relu
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        if self.relu:
            v3 = torch.clamp_max(v2, self.max)
        else:
            v3 = torch.clamp(v2, min=None, max=self.max)
        return v3
relu = False
min = -0.75
max = 1
# Inputs to the model
x1 = torch.randn(1, 2, 32, 32)
