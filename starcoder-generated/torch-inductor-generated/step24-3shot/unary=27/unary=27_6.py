
class Model(torch.nn.Module):
    def __init__(self, min_value=0.1, max_value=0.11):
        super().__init__()
        self.conv = torch.nn.Conv2d(7, 1, 1, stride=2, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 7, 103, 78)
