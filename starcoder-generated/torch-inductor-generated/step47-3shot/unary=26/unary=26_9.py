
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(15, 16, 3, stride=1, padding=1, bias=False)
    def forward(self, x3):
        y1 = self.conv_t(x3)
        y2 = y1 > 69.968
        y3 = y1 * 0.2438
        y4 = torch.where(y2, y1, y3)
        return y2.float()
# Inputs to the model
x3 = torch.randn(34, 15, 45, 14)
