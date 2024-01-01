
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, 1, stride=1, padding=1, bias=False)
    def forward(self, x6):
        z5 = self.conv_t(x6)
        z6 = z5 > 0
        z7 = z5 * -0.175
        z8 = torch.where(z6, z5, z7)
        return torch.nn.functional.interpolate(z8, scale_factor=[1.0, 1.0])
# Inputs to the model
x6 = torch.randn(3, 1, 49, 91)
