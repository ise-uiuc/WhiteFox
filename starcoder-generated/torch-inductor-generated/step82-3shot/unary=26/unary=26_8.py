
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(18, 6, kernel_size=4, bias=False, stride=2, padding=1)
    def forward(self, x18):
        x1 = self.conv_t(x18)
        x2 = x1 > 0
        x3 = x1 * -0.5
        x4 = torch.where(x2, x1, x3)
        x5 = x4 * -1.9
        x6 = torch.sigmoid(x5)
        return x6
# Inputs to the model
x18 = torch.randn(2, 18, 14, 16)
