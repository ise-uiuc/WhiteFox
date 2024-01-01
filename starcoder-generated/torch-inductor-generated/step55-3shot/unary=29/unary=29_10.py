
class Model(torch.nn.Module):
    def __init__(self, min_value=-78, max_value=0):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(channel, size, 1, stride=3, padding=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(size, channel, 1, stride=3, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.conv_transpose(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, channel, 751, 753)
