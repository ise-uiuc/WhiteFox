
class Model(torch.nn.Module):
    def __init__(self, min_value=0.6, max_value=0.85):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 3, stride=7, padding=1)
        self.conv1 = torch.nn.Conv2d(2, 3, 3, stride=5, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = self.conv1(v2)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 22, 39)
