
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(76, 19, 3, stride=4, padding=1, bias=True)
    def forward(self, x3):
        k1 = self.conv_t(x3)
        k2 = k1 > 0
        k3 = k1 * -0.355
        k4 = torch.where(k2, k1, k3)
        return torch.abs(k4)
# Inputs to the model
x3 = torch.randn(2, 76, 3, 36)
