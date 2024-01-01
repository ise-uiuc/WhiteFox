
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(8, 59, 1, stride=1, padding=0, bias=True)
    def forward(self, x47):
        c1 = self.conv_t(x47)
        c2 = c1 > 0.0053
        c3 = c1 * 0.185
        c4 = torch.where(c2, c1, c3)
        return torch.nn.functional.interpolate(c4, size=(44, 56), mode='bilinear', align_corners=None)
# Inputs to the model
x47 = torch.randn(41, 8, 54, 70)
