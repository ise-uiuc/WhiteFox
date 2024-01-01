
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 5, 1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
min_value = 0.01
max_value = 0.0018
# Inputs to the model
x1 = torch.randn(1, 4, 224, 224)
