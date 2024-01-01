
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.5, max_value=0.5):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(8, 24, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv2(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 8, 528, 528)
