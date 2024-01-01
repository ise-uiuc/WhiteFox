
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(780, 248, 3, stride=(7, 1), padding=11, output_padding=5, bias=True)
    def forward(self, x3):
        k1 = self.conv_t(x3)
        k2 = k1 > 0
        k3 = k1 * 0.889
        k4 = torch.where(k2, k1, k3)
        return k4
# Inputs to the model
x3 = torch.randn(8, 780, 89, 6)
