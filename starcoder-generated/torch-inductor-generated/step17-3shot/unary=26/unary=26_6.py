
class Model(torch.nn.Module):
    def __init__(self, negative_slope=1):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 15, (7, 9), 3, 3, 2, 1, 1)
    def forward(self, x):
        x2 = self.conv_t(x)
        x3 = x2 > 0
        x4 = x2 * self.negative_slope
        x5 = torch.where(x3, x2, x4)
        return x5
negative_slope = 1
# Inputs to the model
x = torch.randn(8, 3, 16, 16)
