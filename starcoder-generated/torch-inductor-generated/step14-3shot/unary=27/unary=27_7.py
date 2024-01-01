
class Model(torch.nn.Module):
    def __init__(self, min_value=0.6, max_value=0.7):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.clamp(min=self.min_value)
        v3 = v2.clamp(max=self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
