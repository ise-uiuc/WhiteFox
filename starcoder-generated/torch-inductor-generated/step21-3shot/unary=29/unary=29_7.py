
class Model(torch.nn.Module):
    def __init__(self, min_value=1, max_value=10):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(5, 5, 7)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((23, 19))
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.avg_pool(v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
# Inputs to the model
x1 = torch.randn(1, 5, 23, 19)
