
class Model(torch.nn.Module):
    def __init__(self, in_channels=3, min_value=0, max_value=3):
        super().__init__()
        self.in_channels = in_channels
        self.min_value = min_value
        self.max_value = max_value
        self.conv_transpose = torch.nn.ConvTranspose2d(self.in_channels, 5, 1, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 17, 31)
