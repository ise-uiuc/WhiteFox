
class Model(torch.nn.Module):
    def __init__(self, min_value=-100, max_value=100):
        super().__init__()
        self.clamp = torch.nn.Conv2d(3, 3, 5, stride=1, padding=2)
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 44, 12, stride=1, padding=8)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.clamp(x1)
        v2 = self.conv_transpose(v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
