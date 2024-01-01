
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 5, 11, stride=2, groups=2, bias=False)
    def forward(self, x5):
        v1 = self.conv_t(x5)
        v2 = v1 > 0
        v3 = v1 * -1
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x5 = torch.randn(1, 4, 8, 8)
