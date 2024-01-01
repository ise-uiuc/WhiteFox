
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 3, 6, stride=2, padding=2, output_padding=1, bias=False)
    def forward(self, x2):
        x1 = self.conv_t(x2)
        x2 = x1 > 0
        x3 = x1 * 2.0052
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x2 = torch.randn(63, 1, 22, 18)
