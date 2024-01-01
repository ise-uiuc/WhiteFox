
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(372, 518, 3, stride=2, padding=1, bias=False)
    def forward(self, x16):
        x1 = self.conv_t(x16)
        x2 = x1 > 0
        x3 = x1 * -3.61321
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x16 = torch.randn(3, 372, 3, 65)
