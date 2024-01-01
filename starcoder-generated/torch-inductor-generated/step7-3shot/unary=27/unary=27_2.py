
class Model(torch.nn.Module):
    def __init__(self, min_value=0.5, max_value=0.5):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0)
        self.conv_2 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.conv_2(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
