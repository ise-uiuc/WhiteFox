
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(911, 121, 3, stride=5, padding=0)
        self.negative_slope = negative_slope
    def forward(self, x11):
        p1 = self.conv_t(x11)
        p2 = p1 > 0
        p3 = p1 * self.negative_slope
        p4 = torch.where(p2, p1, p3)
        return p4
negative_slope = 0.084
# Inputs to the model
x11 = torch.randn(13, 911, 41, 63)
