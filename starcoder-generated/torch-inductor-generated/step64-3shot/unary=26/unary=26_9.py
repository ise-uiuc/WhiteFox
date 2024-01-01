
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(10, 3155, 4, stride=2, padding=2, bias=False)
    def forward(self, x6):
        z1 = self.conv_t(x6)
        z2 = z1 > 1
        z3 = z1 * -0.1795
        z4 = torch.where(z2, z1, z3)
        return torch.nn.functional.interpolate(z4, size=7, mode='bilinear', align_corners=True)
# Inputs to the model
x6 = torch.randn(2, 10, 109, 96, dtype=torch.float32, device='cuda')
