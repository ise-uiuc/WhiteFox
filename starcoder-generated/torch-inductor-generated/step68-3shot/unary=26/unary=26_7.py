
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(26, 49, 3, stride=1, padding=3, bias=False)
    def forward(self, x7):
        v7 = self.conv_t(x7)
        v8 = v7 > 0
        v9 = v7 * 0.425
        v10 = torch.where(v8, v7, v9)
        return v10
# Inputs to the model
x7 = torch.randn(2, 26, 24, 31)
