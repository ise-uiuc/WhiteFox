
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(19, 64, 1, stride=1, padding=0)
        self.conv_t2 = torch.nn.ConvTranspose2d(64, 19, 1, stride=1, padding=0)
        self.conv_t3 = torch.nn.ConvTranspose2d(19, 19, 1, stride=1, padding=0)
        self.negative_slope = negative_slope
    def forward(self, x6):
        x7 = self.conv_t1(x6)
        x8 = self.conv_t2(x7)
        x9 = self.conv_t3(x8)
        x10 = x9 > 0
        x11 = x9 * self.negative_slope
        x12 = torch.where(x10, x9, x11)
        return x12
negative_slope = -0.1
# Inputs to the model
x6 = torch.randn(2, 19, 4, 4)
