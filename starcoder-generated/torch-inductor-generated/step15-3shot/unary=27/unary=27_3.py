
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.4, max_value=0.3):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 7, 1, stride=2, padding=3)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        conv = self.conv(x1)
        clamp_min = torch.clamp(conv, min=self.min_value)
        clamp_max = torch.clamp(clamp_min, max=self.max_value)
        return clamp_max
# Inputs to the model
x1 = torch.randn(1, 6, 20, 20)
