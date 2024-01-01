
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(18, 9, 8, stride=1, bias=False)
    def forward(self, x7):
        u2 = self.conv_t(x7)
        v3 = u2 > 0
        v4 = u2 * -0.006321958026499228
        v5 = torch.where(v3, u2, v4)
        return v5
# Inputs to the model
x7 = torch.randn(5, 18, 53, 10)
