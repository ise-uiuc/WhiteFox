
class Model(torch.nn.Module):
    def __init__(self, min_value=272184032, max_value=0):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 3, 1, stride=1, padding=1)
        self.conv = torch.nn.Conv2d(3, 2, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv(v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
