
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.0002, max_value=-0.0001):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(100, 100, 3, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        v1 = self.conv_transpose(x)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(100, 101, 102, 103)
