
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(367, 346, 5, stride=1, padding=2, bias=False)
    def forward(self, x7):
        v1 = self.conv_t(x7)
        v2 = v1 > 0
        v3 = v1 * -1.6417
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x7 = torch.randn(24, 367, 74, 72)
