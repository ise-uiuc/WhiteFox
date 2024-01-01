
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.45, max_value=-1.3):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(6, 15, 1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = torch.clamp_max(x1, self.max_value)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = self.conv2d(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 6, 12, 12)
