
class Model(torch.nn.Module):
    def __init__(self, min_value=0, max_value=0):
        super().__init__()
        self.conv = torch.nn.Conv2d(256, 32, 1, stride=1, padding=0)
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 16, 75, stride=53, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 256, 1, 1)
