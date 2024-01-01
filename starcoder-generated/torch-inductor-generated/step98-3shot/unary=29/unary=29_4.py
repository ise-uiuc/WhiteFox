
class Model(torch.nn.Module):
    def __init__(self, min_value=0.0100, max_value=0.0139):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=3)
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 8, 1, stride=1, padding=1)
        self.clamp = torch.nn.Clamp(min=min_value, max=max_value)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        v3 = torch.clamp(v2, self.min_value, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 9, 9)
