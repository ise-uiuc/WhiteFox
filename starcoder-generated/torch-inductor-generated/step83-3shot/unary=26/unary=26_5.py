
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(32, 32, 5, stride=1, padding=0)
    def forward(self, x2):
        x3 = self.conv_t(x2)
        x4 = x3 > 0
        x5 = x3 * 0.0439
        x6 = torch.where(x4, x3, x5)
        return torch.nn.functional.interpolate(x6, scale_factor=[6.0, 5.0])
# Inputs to the model
x2 = torch.randn(9, 32, 10, 6)
