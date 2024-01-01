
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(7, 1080, 2, stride=1, padding=2, bias=False)
    def forward(self, x2):
        y1 = self.conv_t(x2)
        y2 = y1 > 0
        y3 = y1 * -0.807445
        y4 = torch.where(y2, y1, y3)
        y5 = torch.flatten(y4, 1)
        return y5
# Inputs to the model
x2 = torch.randn(20625, 7, 4, 30)
