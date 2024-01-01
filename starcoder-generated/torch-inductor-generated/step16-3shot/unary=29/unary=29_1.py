
class Model(torch.nn.Module):
    def __init__(self, min_value=-227.0, max_value=233.0):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1)
        self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d(7)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.adaptive_avg_pool2d(v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
