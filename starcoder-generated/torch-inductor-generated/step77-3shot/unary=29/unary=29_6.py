
class Model(torch.nn.Module):
    def __init__(self, min_value=0, max_value=8):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(7, 1, 5, stride=5, padding=0, dilation=4, groups=(1, 1), bias=False, padding_mode='zeros')
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 7, 8, 8)
