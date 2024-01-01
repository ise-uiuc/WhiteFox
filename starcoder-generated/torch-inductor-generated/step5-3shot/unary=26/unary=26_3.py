
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(8, 3, 3, stride=2, padding=1, output_padding=1)
        self.negative_slope = 3
    def forward(self, x1):
        x2 = self.conv_transpose(x1)
        x3 = x2 > 0
        x4 = x2 * self.negative_slope
        x5 = torch.where(x3, x2, x4)
        return x5
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64, 64)
