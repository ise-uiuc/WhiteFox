
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(156, 128, 4, stride=2, padding=1, bias=False)
    def forward(self, x14):
        y1 = self.conv_t(x14)
        y2 = y1 > 0
        y3 = y1 * -1.25
        y4 = torch.where(y2, y1, y3)
        return y4
# Inputs to the model
x14 = torch.randn(32, 156, 84, 73)
