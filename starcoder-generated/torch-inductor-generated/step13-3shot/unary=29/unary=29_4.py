
class Model(torch.nn.Module):
    def __init__(self, min_value=2, max_value=0):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 9, 3)
        self.conv_transpose = torch.nn.ConvTranspose2d(9, 8, 5)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.conv_transpose(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
