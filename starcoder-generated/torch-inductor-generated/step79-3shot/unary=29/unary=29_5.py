
class Model(torch.nn.Module):
    def __init__(self, min_value=0.010218, max_value=0.01845):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 2, 7, stride=2, padding=3, dilation=1, output_padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
