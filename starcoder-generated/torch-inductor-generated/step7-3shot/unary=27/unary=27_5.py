
class Model(torch.nn.Module):
    def __init__(self, min_value=-2.0, max_value=1.5):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=3, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
