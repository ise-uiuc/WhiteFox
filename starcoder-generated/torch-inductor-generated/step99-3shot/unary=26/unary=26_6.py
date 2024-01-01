
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(5, 113, 9, stride=2, padding=6, bias=False)
    def forward(self, x0):
        b1 = self.conv_t(x0)
        b2 = b1 > 0
        b3 = b1 * -0.814
        b4 = torch.where(b2, b1, b3)
        return b4
# Inputs to the model
x0 = torch.randn(1, 5, 65, 22)
