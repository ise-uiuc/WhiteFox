
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.avgpool = torch.nn.AvgPool2d(3, stride=1, padding=1)
        self.linear = torch.nn.Linear(10, 16)
        self.min = min
        self.max = max
    def forward(self, x1, x2, x3):
        v1 = self.avgpool(x1)
        v2 = self.linear(x2)
        v3 = torch.max(v1, v2)
        v4 = torch.clamp_min(v3, self.min)
        v5 = torch.clamp_max(v4, self.max)
        return v5, v2, x3
min = -6
max = 7.2
# Inputs to the model
x1 = torch.randn(1, 16, 100, 100)
x2 = torch.randn(1, 10)
x3 = torch.randn(1, 3, 32, 32)
