
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose3d(4, 64, 4, stride=2, padding=2, bias=False, dilation=3)
    def forward(self, x2):
        y1 = self.conv_t(x2)
        y2 = y1 > 0.0
        y3 = y1 * -0.045
        y4 = torch.where(y2, y1, y3)
        return torch.nn.functional.avg_pool3d(y4, (1, 1, 1))
# Inputs to the model
x2 = torch.randn(2, 4, 35, 42, 76)
