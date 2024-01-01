
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.max1 = torch.nn.MaxPool2d(7, stride=5, padding=2)
        self.max2 = torch.nn.MaxPool2d(5, stride=3, padding=2)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.max2
        v2 = self.max1
        v3 = v1(self.max2(x1))
        v4 = v2(self.max2(x1))
        v5 = torch.clamp_min(v3, self.min)
        v6 = torch.clamp_max(v5, self.max)
        return 2 * v6
min = 0.365801
max = 1.5579
# Inputs to the model
x1 = torch.randn(1, 8, 58, 58)
