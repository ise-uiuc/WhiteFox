
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, 5, stride=3, padding=2)
    def forward(self, x3):
        x4 = self.conv_t(x3)
        x5 = torch.flatten(x4, 1)
        x6 = x5 * 1.1
        x7 = x6 / 3.0
        return torch.round(x7)
# Inputs to the model
x3 = torch.randn(1, 1, 56, 110)
