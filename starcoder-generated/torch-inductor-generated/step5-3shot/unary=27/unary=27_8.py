
class Model(torch.nn.Module):
    def __init__(self, min_value=1, max_value=0.3):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 9, 2, stride=3, padding=7)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x0):
        v1 = self.conv(x0)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x0 = torch.randn(1, 3, 64, 64)
