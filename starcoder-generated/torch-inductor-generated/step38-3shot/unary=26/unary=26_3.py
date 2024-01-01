
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(7, 13, 7, padding=5, stride=2)
    def forward(self, x):
        y1 = self.conv_t(x)
        y2 = y1 > 0
        y3 = y1 * -0.131
        y4 = torch.where(y2, y1, y3)
        return y4
# Inputs to the model
x = torch.randn(31, 7, 91, 67)
