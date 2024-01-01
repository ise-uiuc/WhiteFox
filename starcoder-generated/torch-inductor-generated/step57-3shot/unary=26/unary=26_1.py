
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(256, 64, 2, stride=2, padding=1, output_padding=1, bias=False)
    def forward(self, x11):
        x1 = self.conv_t(x11)
        x2 = x1 > 0
        x3 = x1 * -7.11
        x4 = torch.where(x2, x1, x3)
        return torch.nn.functional.avg_pool2d(x4, 2)
# Inputs to the model
x11 = torch.randn(4, 256, 150, 111)
