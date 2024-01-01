
class Model(torch.nn.Module):
    def __init__(self, min_value=0, max_value=0):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 3, 232, stride=1, padding=1, dilation=1, output_padding=0, groups=1, bias=False, padding_mode='zeros')
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.relu6(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 281, 281)
