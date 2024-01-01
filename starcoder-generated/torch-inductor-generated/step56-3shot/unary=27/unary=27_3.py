
class Model(torch.nn.Module):
    def __init__(self, min_value):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.min_value = min_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        return v2
min_value = 0.7
# Inputs to the model
x1 = torch.randn(1, 1, 100, 200)
# Model begins