
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(426, 75, 2, stride=1, padding=0)
    def forward(self, x1):
        b1 = self.conv_t(x1)
        b2 = b1 > 0
        b3 = b1 * -3.7029
        b4 = torch.where(b2, b1, b3)
        return b4
# Inputs to the model
x1 = torch.randn(3, 426, 141, 82)
