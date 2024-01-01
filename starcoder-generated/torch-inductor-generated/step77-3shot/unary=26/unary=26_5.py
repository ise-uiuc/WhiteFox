
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(7, 851, kernel_size=(5,8), stride=3, padding=0, groups=4)
    def forward(self, x):
        y1 = self.bn_f(x)
        y2 = self.conv_t(y1)
        y3 = y2 > 0
        y4 = y2 * 1.427
        y5 = torch.where(y3, y2, y4)
        return y5
# Inputs to the model
x = torch.randn(1, 7, 18, 19)
