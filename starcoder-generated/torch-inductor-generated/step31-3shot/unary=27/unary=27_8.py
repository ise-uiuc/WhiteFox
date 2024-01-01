
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv1d(13, 10, 11, stride=3, dilation=2)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = torch.nn.functional.avg_pool2d(x1, 3, stride=2, padding=2)
        v2 = self.conv(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 0.6
max = 1.6
# Inputs to the model
x1 = torch.randn(1, 13, 100)
