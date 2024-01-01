
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=7, stride=(2, 2), padding=(2, 2), bias=False)
        self.bn = torch.nn.BatchNorm2d(num_features=1, eps=0.0010000000474974513)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 0.0
max = float("inf")
# Inputs to the model
x1 = torch.randn(1, 3, 65, 65)
