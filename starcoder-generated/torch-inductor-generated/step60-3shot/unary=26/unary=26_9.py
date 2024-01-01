
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(35, 35, 5, bias=False)
    def forward(self, x19):
        i1 = self.conv_t(x19)
        i2 = i1 > 0
        i3 = i1 * -0.229
        i4 = torch.where(i2, i1, i3)
        return torch.nn.functional.interpolate(torch.nn.ReLU()(i4), scale_factor=2.863, recompute_scale_factor=True)
# Inputs to the model
x19 = torch.randn(31, 35, 50, 4, device='cpu')
