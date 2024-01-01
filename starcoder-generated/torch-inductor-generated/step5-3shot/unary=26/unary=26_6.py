
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1)
        self.negative_slope = negative_slope
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        x2 = self.conv(x1)
        x3 = x2 > 0
        x4 = x2 * self.negative_slope
        x5 = torch.where(x3, x2, x4)
        return x5
negative_slope = 1
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
