
class Model(torch.nn.Module):
    def __init__(self, min_value=-2.0, max_value=12.0):
        super().__init__()
        self.conv_transpose3d = torch.nn.ConvTranspose3d(3, 2, kernel_size=(1, 1, 1))
        self.conv2d = torch.nn.Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1))
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x8):
        v9 = self.conv_transpose3d(x8)
        v10 = self.conv2d(v9)
        v11 = torch.clamp_min(v10, self.min_value)
        v12 = torch.clamp_max(v11, self.max_value)
        return v12
# Inputs to the model
x8 = torch.randn(16, 3, 4, 36, 56)
