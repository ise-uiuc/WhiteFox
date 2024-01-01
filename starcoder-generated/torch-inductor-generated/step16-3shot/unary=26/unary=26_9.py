
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(9, 10, 4, stride=1, padding=4)
    def forward(self, x0):
        y1 = self.conv_t(x0)
        y2 = y1 > 0
        y3 = y1 % 2
        y4 = torch.where(y2, y3, y1)
        return y4
# Inputs to the model
x0 = torch.randn(8, 9, 15, 17)
