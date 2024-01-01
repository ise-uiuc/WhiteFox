
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 2, stride=2, padding=5)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
min_value = -0.86
max_value = -0.41
# Inputs to the model
x = torch.randn(3, 1, 65, 65)
