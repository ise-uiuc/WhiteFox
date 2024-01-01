
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(6, 30, 5, stride=1, padding=0)
        self.conv_2 = torch.nn.Conv2d(30, 6, 5, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = self.conv_2(v2)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = -2
max = 2
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
