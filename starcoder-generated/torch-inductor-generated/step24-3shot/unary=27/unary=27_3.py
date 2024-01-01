
class Model(torch.nn.Module):
    def __init__(self, min_value=0.1, max_value=0.11):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 2, stride=1, padding=4, dilation=2)
        self.convbatch = torch.nn.BatchNorm2d(8)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.convbatch(v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
