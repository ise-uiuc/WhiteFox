
class Model(torch.nn.Module):
    def __init__(self, min_value=0.26, max_value=0.5):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 70, 80)
