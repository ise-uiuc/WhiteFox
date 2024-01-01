
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 5, 5, dilation=2, stride=3, padding=1)
        self.conv1 = torch.nn.Conv2d(5, 6, 5, dilation=1, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x1, x2):
        v1 = torch.clamp_min(x1, self.min)
        v2 = torch.clamp_max(x2, self.max)
        v4 = self.conv(v1)
        v3 = self.conv1(v4 + v2)
        return v3
min = 0.7
max = 0.7
# Inputs to the model
x1 = torch.randn(4, 4, 10)
x2 = torch.randn(1, 5, 15)
