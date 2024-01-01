
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 128, 7, stride=2, padding=2, bias=False)
    def forward(self, x2):
        a1 = self.conv_t(x2)
        a2 = a1 > 0
        a3 = a1 * -4.94
        a4 = torch.where(a2, a1, a3)
        return a4
# Inputs to the model
x2 = torch.randn(4, 4, 35, 42)
