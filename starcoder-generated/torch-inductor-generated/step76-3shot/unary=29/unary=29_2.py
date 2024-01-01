
class Model(torch.nn.Module):
    def __init__(self, min_value=-234.3169, max_value=293.1789):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(1, 17, 1, stride=2, padding=1, output_padding=2)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(3, 1, 8, 3, 7)
