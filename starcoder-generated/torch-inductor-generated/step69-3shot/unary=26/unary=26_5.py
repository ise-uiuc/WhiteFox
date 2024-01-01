
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 8, kernel_size=2, stride=2, padding=0, output_padding=1, bias=False)
    def forward(self, x1):
        y1 = self.conv_t(x1)
        y2 = y1 > 0
        y3 = y1 * 1
        y4 = torch.where(y2, y1, y3)
        return y4
# Inputs to the model
x1 = torch.randn(4, 4, 4, 4)
