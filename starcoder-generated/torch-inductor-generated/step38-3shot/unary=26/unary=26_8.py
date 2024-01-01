
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 348, 3, stride=1, padding=1, bias=False)
    def forward(self, x):
        y1 = self.conv_t(x)
        y2 = y1 > 0
        y3 = y1 * 0.148
        y4 = torch.where(y2, y1, y3)
        return y4
# Inputs to the model
x = torch.randn(3, 4, 35, 90)
