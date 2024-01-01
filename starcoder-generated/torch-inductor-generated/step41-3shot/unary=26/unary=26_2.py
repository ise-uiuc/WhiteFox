
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(76, 56, 4, stride=1, padding=0, bias=False)
    def forward(self, x6):
        v1 = self.conv_t(x6)
        v2 = v1 > 0
        v3 = v1 * 0.00078
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x6 = torch.randn(655, 76, 11, 12)
