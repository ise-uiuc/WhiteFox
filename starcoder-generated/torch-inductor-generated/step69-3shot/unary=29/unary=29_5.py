
class Model(torch.nn.Module):
    def __init__(self, min_value=0.5789, max_value=0.8308):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.conv = torch.nn.Conv2d(26, 26, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 26, 31, 26)
