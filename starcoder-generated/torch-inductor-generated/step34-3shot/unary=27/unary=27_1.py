
class Model(torch.nn.Module):
    def __init__(self, min_value=0.3, max_value=0.5):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, t):
        v1 = self.conv(t)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
