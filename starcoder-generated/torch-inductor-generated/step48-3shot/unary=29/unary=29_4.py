
class Model(torch.nn.Module):
    def __init__(self, min_value=1.98, max_value=1.58):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 1, stride=2, dilation=2, groups=1, bias=True, padding=3, output_padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 2, 3)
