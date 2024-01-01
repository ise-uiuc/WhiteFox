
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(64, 3, 12, stride=2, padding=7, bias=False)
    def forward(self, x1):
        y1 = self.conv_t(x1)
        y2 = y1 > 0
        y3 = y1 * -7.604
        y4 = torch.where(y2, y1, y3)
        return y4
# Inputs to the model
x1 = torch.randn(3, 64, 7, 8, requires_grad=False)
