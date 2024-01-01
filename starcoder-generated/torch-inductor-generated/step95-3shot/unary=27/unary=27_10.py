
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 46, 3, stride=1, padding=4, dilation=5)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = -68.25299224853516
max = -32.71911811828613
# Inputs to the model
x1 = torch.randn(1, 1, 10, 10)
