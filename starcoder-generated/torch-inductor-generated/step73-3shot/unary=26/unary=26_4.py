
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2, 35, 11, stride=1, padding=9, bias=False)
    def forward(self, x41):
        x1 = self.conv_t(x41)
        x2 = x1 > 0
        x3 = x1 * -0.903
        x4 = torch.where(x2, x1, x3)
        return torch.nn.functional.interpolate(torch.nn.ReLU()(x4), (25, 37))
# Inputs to the model
x41 = torch.randn(2, 2, 49, 36)
