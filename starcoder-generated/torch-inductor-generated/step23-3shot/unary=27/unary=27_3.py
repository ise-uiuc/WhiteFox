
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.avgpool = torch.nn.AvgPool2d(3, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = self.avgpool(t1)
        t3 = torch.clamp_min(t2, self.min_value)
        t4 = torch.clamp_max(t3, self.max_value)
        return t4

min_value = 0.77
max_value = 0.8
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
