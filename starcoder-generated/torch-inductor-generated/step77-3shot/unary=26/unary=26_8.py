
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 128, 7, stride=3, padding=0, dilation=3, groups=1, bias=False)
    def forward(self, x12):
        v9 = self.conv_t(x12)
        v10 = v9 > 0
        v11 = v9 * -83.51
        v12 = torch.where(v10, v9, v11)
        return v12
# Inputs to the model
x12 = torch.randn(15, 3, 29, 16)
