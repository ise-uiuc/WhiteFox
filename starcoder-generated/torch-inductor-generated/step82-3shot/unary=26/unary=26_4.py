
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 3, 3)
    def forward(self, x19):
        x8 = self.conv_t(x19)
        x9 = x8 < 0
        x10 = x8 > 0
        x11 = ~x9
        x12 = x11 * -0.899287
        x13 = torch.where(x10, x8, x12)
        x14 = torch.neg(x13)
        return torch.floor(x14)
# Inputs to the model
x19 = torch.randn(1, 3, 3, 15)
