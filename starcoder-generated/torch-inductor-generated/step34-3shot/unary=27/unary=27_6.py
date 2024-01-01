
class Model(torch.nn.Module):
    def __init__(self, min_value):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
        self.min_value = min_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        return v2
min_value = 10
# Inputs to the model
x1 = torch.randn(1, 3, 100, 100)
