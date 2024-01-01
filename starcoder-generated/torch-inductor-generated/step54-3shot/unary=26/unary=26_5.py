
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2, 204, 15, stride=11, padding=7, bias=True)
    def forward(self, x10):
        x1 = self.conv_t(x10)
        x2 = x1 > 0
        x3 = x1 * -1.4
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x10 = torch.randn(3, 2, 23, 102)
