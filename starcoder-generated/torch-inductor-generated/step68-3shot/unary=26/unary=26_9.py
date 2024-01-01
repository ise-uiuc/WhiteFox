
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(12, 16, 3, stride=1, padding=3, bias=False)
    def forward(self, x5):
        v1 = self.conv_t(x5)
        v4 = torch.nn.functional.interpolate(v1, scale_factor=[1.875, 2.0125])
        return v4
# Inputs to the model
x5 = torch.randn(9, 12, 193, 92)
