
class Model(torch.nn.Module):
    def __init__(self, min_value=8.4827, max_value=-3.5561):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 4, (1, 6), stride=2, groups=2, padding=0, dilation=3, padding_mode='zeros')
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 81, 40)
