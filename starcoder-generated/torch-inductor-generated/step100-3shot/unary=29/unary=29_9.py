
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.465224797482, max_value=0.513743235683):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 197, 11, stride=1, padding=1)
        self.conv = torch.nn.Conv2d(197, 29, 1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.conv(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 1, 1)
