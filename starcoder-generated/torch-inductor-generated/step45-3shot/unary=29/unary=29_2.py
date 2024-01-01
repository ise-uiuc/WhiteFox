
class Model(torch.nn.Module):
    def __init__(self, min_value=-6, max_value=8):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 5, 2, stride=8, output_padding=12, padding=7)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 99, 98)
