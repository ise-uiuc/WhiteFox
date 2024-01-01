
class Model(torch.nn.Module):
    def __init__(self, min_value=0.0099, max_value=0.6193):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 98, 1, stride=1, padding=0)
        self.conv_transpose = torch.nn.ConvTranspose2d(10, 98, 75, stride=3, padding=0)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(10, 98, 3, stride=2, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        v3 = self.conv_transpose1(v2)
        v4 = torch.clamp_min(v3, self.min_value)
        v5 = torch.clamp_max(v4, self.max_value)
        return v5
# Inputs to the model
x1 = torch.randn(1, 10, 2, 2)
