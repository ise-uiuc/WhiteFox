
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(6, 89, 9, stride=2, bias=False)
    def forward(self, x11):
        v1 = self.conv_t(x11)
        v2 = v1 > 0
        v3 = v1 * -7.13
        v4 = torch.where(v2, v1, v3)
        return torch.nn.functional.pixel_shuffle(v4, 12)
# Inputs to the model
x11 = torch.randn(1, 6, 67, 56)
