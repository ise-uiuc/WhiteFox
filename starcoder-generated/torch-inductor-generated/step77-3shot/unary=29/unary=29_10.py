
class Model(torch.nn.Module):
    def __init__(self, min_value=-26.1419, max_value=-315.711937):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(11, 16, 3, stride=(2, 1), padding=(4, 1), dilation=3, output_padding=(1, 1))
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Input to the model
x1 = torch.randn(1, 11, 26, 73)
