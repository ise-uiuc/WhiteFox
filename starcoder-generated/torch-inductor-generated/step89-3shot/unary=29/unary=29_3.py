
class Model(torch.nn.Module):
    def __init__(self, min_value=1161.9418, max_value=1328.7794):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 7, 3, stride=2, padding=1, output_padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 4, 7)
