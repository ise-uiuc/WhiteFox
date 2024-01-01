
class Model(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(26, 19, kernel_size=kernel_size, stride=1, padding=0)
        self.negative_slope = 0.0001
    def forward(self, x2):
        v1 = self.conv_t(x2)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
kernel_size = (2, 2)
# Inputs to the model
x2 = torch.randn(16, 26, 8, 8)
