
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1427, 348, 7, stride=1, padding=0, dilation=1, groups=1, bias=False)
    def forward(self, x):
        xe = self.conv_t(x)
        xf = xe > 0
        xg = xe * 0.17
        xc = torch.where(xf, xe, xg)
        return xc
# Inputs to the model
x = torch.randn(10, 1427, 46, 24)
