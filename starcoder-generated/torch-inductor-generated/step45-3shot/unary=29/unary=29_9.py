
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.4200000047683716, max_value=1.25):
        super().__init__()
        self.avg_pool = torch.nn.AvgPool2d(2)
        self.conv = torch.nn.Conv2d(2, 1, 4, 1, 2)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.avg_pool(x1)
        v2 = self.conv(v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 3, 5)
