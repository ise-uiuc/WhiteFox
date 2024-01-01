
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(5, 124, 3, stride=2, padding=1, bias=False)
    def forward(self, x5):
        k1 = self.conv_t(x5)
        k2 = k1 > 4.964
        k3 = k1 * 0.214
        k4 = torch.where(k2, k1, k3)
        return k4
# Inputs to the model
x5 = torch.randn(9, 5, 109, 42)
