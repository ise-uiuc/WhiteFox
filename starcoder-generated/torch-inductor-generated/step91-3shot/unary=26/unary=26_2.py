
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(10, 6, 3, stride=1, padding=1, bias=False)
    def forward(self, x4):
        v1 = self.conv_t(x4)
        v2 = v1 > 0
        v3 = v1 * -0.5521483
        v4 = torch.where(v2, v1, v3)
        return v2
# Inputs to the model
x4 = torch.randn(23, 10, 8, 54)
