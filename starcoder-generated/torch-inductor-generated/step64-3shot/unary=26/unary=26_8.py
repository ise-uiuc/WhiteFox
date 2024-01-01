
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(286, 900, 7, stride=1, padding=0, bias=False)
    def forward(self, x2):
        k1 = self.conv_t(x2)
        k2 = k1 > 0
        k3 = k1 * -29.9274
        k4 = torch.where(k2, k1, k3)
        return k4
# Inputs to the model
x2 = torch.randn(9, 286, 21, 95)
