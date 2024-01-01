
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(622, 375, 1, stride=1, padding=0, bias=False)
    def forward(self, x75):
        v19 = self.conv_t(x75)
        v20 = v19 > 0
        v21 = v19 * 1.1879
        v22 = torch.where(v20, v19, v21)
        return v22
# Inputs to the model
x75 = torch.randn(18, 622, 82, 66)
