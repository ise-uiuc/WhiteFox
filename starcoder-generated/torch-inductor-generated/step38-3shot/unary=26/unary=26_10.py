
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 3, 3, stride=1, padding=2, bias=False)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * -1.403
        x4 = torch.where(x2, x1, x3)
        return torch.nn.functional.interpolate(x4, scale_factor=[1.0, 1.0])
# Inputs to the model
x = torch.randn(4, 3, 9, 8)
