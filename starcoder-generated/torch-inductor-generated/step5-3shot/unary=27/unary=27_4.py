
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=3)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
min_value = 0.9
max_value = 0.9
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
