
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(765, 8, 1, stride=1, padding=0, bias=False)
    def forward(self, x3):
        y1 = self.conv_t(x3)
        y2 = y1 > 0
        y3 = y1 * 0.4637
        y4 = torch.where(y2, y1, y3)
        return torch.nn.functional.interpolate(y4, scale_factor=[6.0, 5.0])
# Inputs to the model
x3 = torch.randn(33, 765, 8, 3)
