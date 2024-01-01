
class Model(torch.nn.Module):
    def __init__(self, min_value=475.75, max_value=489.0):
        super().__init__()
        self.threshold = torch.nn.Threshold(0.0, 0.0)
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 6, 2, stride=2, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x2):
        v1 = self.conv_transpose(x2)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.threshold(v3, 0.0)
        return v4
# Inputs to the model
x2 = torch.randn(1, 8, 19, 17)
