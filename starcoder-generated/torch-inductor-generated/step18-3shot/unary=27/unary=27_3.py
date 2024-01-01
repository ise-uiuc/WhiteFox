
class Model(torch.nn.Module):
    def __init__(self, min_value=-3.0, max_value=0.928):
        super().__init__()
        self.conv = torch.nn.Conv1d(33, 87, 1, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 33, 33)
