
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(19, 64, 1, stride=1, padding=0, groups=7)
    def forward(self, x2):
        v1 = self.conv_t(x2)
        v2 = v1 > 0
        v3 = v1 * -0.25
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x2 = torch.randn(8, 19, 4, 4)
