
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(102, 2000, kernel_size==[1, (2, 5), 1], stride=[2, 4, 1], padding=([3, 0], (2, 3,.5), 5))
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = -2648.4251132903843
max = -649.1555929390952
# Inputs to the model
x1 = torch.randn(5, 102, 8, 3)
