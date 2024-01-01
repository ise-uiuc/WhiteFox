
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t_1 = torch.nn.ConvTranspose2d(480, 7, 2, stride=2)
    def forward(self, x2):
        y = self.conv_t_1(x2)
        y0 = y > 0.83
        y1 = y * 2.468
        y2 = torch.where(y0, y, y1)
        y3 = y * 3.463
        y4 = torch.where(y0, y3, y2)
        return y4
# Inputs to the model
x2 = torch.randn(8, 480, 8, 8)
