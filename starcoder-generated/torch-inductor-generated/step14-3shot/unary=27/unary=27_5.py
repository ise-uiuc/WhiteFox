
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1, stride=1, padding=1, groups=3, bias=False)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
min_value = 0.2
max_value = 0.4
# Inputs to the model
x1 = torch.randn(1,3,64,64)
