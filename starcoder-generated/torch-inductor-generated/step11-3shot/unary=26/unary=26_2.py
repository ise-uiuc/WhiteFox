
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(11, 3, 2, stride=2)
        self.conv_t3 = torch.nn.ConvTranspose3d(22, 44, 3)
    def forward(self, x2):
        x5 = self.conv_t(x2)
        x6 = x5 * 0.578
        x7 = self.conv_t3(x2)
        return x6, x7
# Inputs to the model
x2 = torch.randn(3, 11, 8, 8)
