
class Model(torch.nn.Module):
    def __init__(self, min_value=-255, max_value=44):
        super().__init__()
        self.conv = torch.nn.Conv2d(128, 20, 5, stride=1, padding=2, dilation=1)
        self.tanh = torch.nn.Tanh()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.tanh(v3)
        return v4
# Input to the model
x1 = torch.randn(2, 128, 480, 640)
