
class Model(torch.nn.Module):
    def __init__(self, min_value=0.592785640239, max_value=0.0528118573668):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1828, 526, 1, stride=3, padding=0)
        self.conv = torch.nn.Conv2d(508, 783, 1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.conv(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1828, 35, 35)
