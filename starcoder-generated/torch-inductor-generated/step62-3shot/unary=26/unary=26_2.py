
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 2, 4, stride=2, padding=2, output_padding=1)
        self.conv_t1 = torch.nn.ConvTranspose2d(2, 2, 1, stride=2, padding=0)
        self.conv_t2 = torch.nn.ConvTranspose2d(2, 2, 1, stride=2, padding=0)
        self.conv_t3 = torch.nn.ConvTranspose2d(2, 2, 1, stride=2, padding=1, dilation=2,output_padding=0)
        self.negative_slope = negative_slope
    def forward(self, x4):
        h1 = self.conv1(x4)
        h2 = h1 > 0
        h3 = h1 * self.negative_slope
        h4 = torch.where(h2, h1, h3)
        h5 = self.conv_t1(h4)
        h6 = h5 > 0
        h7 = h5 * self.negative_slope
        h8 = torch.where(h6, h5, h7)
        h9 = self.conv_t2(h8)
        h10 = h5 > 0
        h11 = h5 * self.negative_slope
        h12 = torch.where(h10, h5, h11)
        h13 = h5 > 0
        h14 = h5 * self.negative_slope
        h15 = torch.where(h13, h5, h14)
        h16 = self.conv_t3(h15)
        h17 = h5 > 0
        h18 = h5 * self.negative_slope
        h19 = torch.where(h17, h5, h18)
        return torch.nn.functional.interpolate(h19, scale_factor=1.2)
negative_slope = -0.22
# Inputs to the model
x4 = torch.randn(1, 2, 36, 24)
