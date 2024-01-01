
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(256, 17, 4, stride=2, padding=2, bias=False)
    def forward(self, x0):
        x1 = self.conv_t(x0)
        x2 = x1 > 0
        x3 = x1 * 0.195
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x0 = torch.randn(1, 256, 14, 17)
