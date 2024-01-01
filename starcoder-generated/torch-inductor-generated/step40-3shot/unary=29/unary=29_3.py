
class Model(torch.nn.Module):
    def __init__(self, min_value=0, max_value=0):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(2, 2, 2, stride=1, dilation=1, padding=0, groups=1, bias=True)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose2d(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 1, 1)
