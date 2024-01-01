
class Model(torch.nn.Module):
    def __init__(self, min_value=-1.042, max_value=3.0105):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(113, 40, 48, stride=32, padding=19)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 113, 222, 65)
