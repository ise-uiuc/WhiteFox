
class Model(torch.nn.Module):
    def __init__(self, min_value=-2.0, max_value=-1.0):
        super().__init__()
        self.conv_transpose3d = torch.nn.ConvTranspose3d(24, 64, kernel_size=7, stride=2, padding=0, output_padding=0, padding_mode='zeros', dilation=1, groups=24)
        self.t2 = torch.nn.Tanh()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose3d(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = v3 * 0.9891531052589417
        v5 = v4 + 1.035398082256317
        v6 = self.t2(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 24, 80, 88, 10)
