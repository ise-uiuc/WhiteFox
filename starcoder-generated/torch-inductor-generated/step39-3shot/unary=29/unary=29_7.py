
class Model(torch.nn.Module):
    def __init__(self, min_value=5, max_value=10):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=1, bias=False, groups=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 2, stride=1, padding=1, groups=2)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.conv_transpose(v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 5, 5)
