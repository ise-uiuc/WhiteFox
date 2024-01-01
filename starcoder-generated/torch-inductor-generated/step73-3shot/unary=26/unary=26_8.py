
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(250, 50, 5, stride=1, padding=0, bias=False)
    def forward(self, x15):
        i1 = self.conv_t(x15)
        i2 = i1 > 0
        i3 = i1 * -0.312
        i4 = torch.where(i2, i1, i3)
        return torch.nn.functional.interpolate(torch.nn.functional.relu(i4), scale_factor=1.65, recompute_scale_factor=True)
# Inputs to the model
x15 = torch.randn(1, 250, 13, 28)
