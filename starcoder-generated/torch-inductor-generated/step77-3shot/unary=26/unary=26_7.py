
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(622, 690, 10, stride=3, padding=3, dilation=1, groups=1, bias=True)
    def forward(self, x32):
        v9 = self.conv_t(x32)
        v10 = v9 > 0
        v11 = v9 * 0.1237
        v12 = torch.where(v10, v9, v11)
        return v12
# Inputs to the model
x32 = torch.randn(11, 622, 91, 76)
