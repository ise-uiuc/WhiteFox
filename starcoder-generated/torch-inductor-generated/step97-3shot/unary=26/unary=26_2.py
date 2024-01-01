
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2, 2, 3, stride=2, bias=False)
    def forward(self, x9):
        x7 = self.conv_t(x9)
        x8 = x7 > 0
        x9 = x7 * 0.18
        x10 = torch.where(x8, x7, x9)
        x11 = torch.nn.functional.max_pool2d(x10, 5, stride=5)
        return x11
# Inputs to the model
x9 = torch.randn(3, 2, 32, 28)
