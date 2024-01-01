
class Model(torch.nn.Module):
    def __init__(self, min_value=-4.042531001996488, max_value=4.048737071724017):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 2, stride=2, padding=2, dilation=1)
        self.max_value = max_value
        self.min_value = min_value
    def forward(self, input_image5):
        v1 = self.conv_transpose(input_image5)
        v2 = torch.clamp_max(v1, self.max_value)
        v3 = torch.clamp_min(v2, self.min_value)
        return v3
# Inputs to the model
input_image5 = torch.randn(1, 3, 64, 64)
