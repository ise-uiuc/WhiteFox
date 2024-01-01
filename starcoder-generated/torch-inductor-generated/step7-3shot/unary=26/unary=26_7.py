
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, 2, stride=2, padding=0, groups=1)
    def forward(self, x2):
        v1 = self.conv_t(x2)
        v2 = v1 > 0
        v3 = v1 * -2.2
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x2 = torch.randn(1, 1, 3, 3)
