
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(330, 427, 4, stride=2, padding=1, bias=False)
    def forward(self, x16):
        v1 = self.conv_t(x16)
        v2 = v1 > 0
        v3 = v1 * -0.08
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x16 = torch.randn(5, 330, 40, 48)
