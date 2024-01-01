
class Model(torch.nn.Module):
    def __init__(self, min_value=0.2283, max_value=0.2616):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 4, 5, stride=2, padding=3)
        self.conv = torch.nn.Conv2d(4, 2, 1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.conv(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 65, 68)
