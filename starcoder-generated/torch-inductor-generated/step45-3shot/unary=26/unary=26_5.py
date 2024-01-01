
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(61, 76, 3, padding=1, bias=False)
    def forward(self, x1):
        e1 = self.conv_t(x1)
        e2 = e1 > -37.5
        e3 = e1 * 0.22
        e4 = torch.where(e2, e1, e3)
        return e4
# Inputs to the model
x1 = torch.randn(1, 61, 16, 16)
