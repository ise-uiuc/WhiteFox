
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(5, 124, 3, stride=1, padding=2, bias=False)
    def forward(self, x2):
        y1 = self.conv_t(x2)
        y2 = y1 > 0
        y3 = y1 * 0.094
        y4 = torch.where(y2, y1, y3)
        return y4
# Inputs to the model
x2 = torch.randn(2, 5, 95, 66)
