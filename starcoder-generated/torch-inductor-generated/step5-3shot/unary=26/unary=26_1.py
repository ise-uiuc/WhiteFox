
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.negative_slope = 0.01
    def forward(self, x1, x2):
        v1 = (self.conv_transpose1(x1) + self.conv_transpose2(x2))
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
