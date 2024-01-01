
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(8, 36, 1, stride=1, bias=False)
    def forward(self, x):
        y1 = self.conv_t(x)
        y2 = y1 > 0
        y3 = y1 * -0.126
        y4 = torch.where(y2, y1, y3)
        return y4
# Inputs to the model
x = torch.randn(30, 8, 35, 22)
