
class Model(torch.nn.Module):
    def __init__(self, min_value=0.5528, max_value=0.5528):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 2, 3, stride=2, dilation=1, padding=11, output_padding=2)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 22, 15)
