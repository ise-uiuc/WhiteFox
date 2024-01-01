
class Model(torch.nn.Module):
    def __init__(self, min_value=0.45, max_value=0.1):
        super().__init__()
        self.conv = torch.nn.Conv2d(18, 3, 1, stride=3, padding=5)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, input, x1):
        v1 = self.conv(input)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3

min_value = -1
max_value = 2
# Inputs to the model
x1 = torch.randn(1, 18, 64, 64)
