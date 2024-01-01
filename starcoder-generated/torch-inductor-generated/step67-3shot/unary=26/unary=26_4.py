
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_i = torch.nn.Conv2d(111, 82, (1, 6), stride=1, padding=(0, 2), bias=True)
        self.conv_t = torch.nn.ConvTranspose2d(82, 111, 7, stride=1, padding=3, bias=False)
    def forward(self, x2, x3):
        y1 = torch.nn.functional.leaky_relu(x3, negative_slope=0.1, inplace=True)
        y2 = self.conv_i(y1)
        y3 = y2 > 0
        y4 = y2 * 4.11
        y5 = torch.where(y3, y2, y4)
        y6 = self.conv_t(y5)
        y7 = y6 > 0
        y8 = y6 * 1.8342
        y9 = torch.where(y7, y6, y8)
        return y9
# Inputs to the model
x2 = torch.randn(4, 111, 19, 68)
x3 = torch.randn(4, 82, 117, 95)
