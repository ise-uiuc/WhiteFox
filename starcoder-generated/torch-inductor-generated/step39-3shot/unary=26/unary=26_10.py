
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(377, 473, 3, stride=1, padding=0, groups=377, bias=False)
    def forward(self, x13):
        x1 = self.conv_t(x13)
        x2 = x1 > 0
        x3 = x1 * 0.135
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x13 = torch.randn(1, 377, 54, 74)
