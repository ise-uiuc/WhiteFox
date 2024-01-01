
class Model(torch.nn.Module):
    def __init__(self, min_value=11, max_value=16):
        super().__init__()
        self.conv2d = torch.nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        convolution = self.conv2d(x1)
        clamp_min = torch.clamp(convolution, min=self.min_value)
        clamp_max = torch.clamp(clamp_min, max=self.max_value)
        return clamp_max
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
