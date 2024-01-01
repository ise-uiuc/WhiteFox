
class Model(torch.nn.Module):
    def __init__(self, min_value=2.202, max_value=2.260):
        super().__init__()
        self.conv_transpose = torch.nn.Conv2d(3, 1, 2, stride=3, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = torch.clamp_min(x1, self.min_value)
        v2 = torch.clamp_max(v1, self.max_value)
        v3 = self.conv_transpose(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 5, 3)
