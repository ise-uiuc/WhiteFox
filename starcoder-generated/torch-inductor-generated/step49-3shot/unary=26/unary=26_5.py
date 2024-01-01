
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.modules.conv.ConvTranspose3d(68, 58, 2, stride=2, padding=1, bias=False)
    def forward(self, x3):
        x1 = self.conv_t(x3)
        x2 = x1 > 0.332
        x3 = x1 * -0.061
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x3 = torch.randn(4, 68, 51, 43, 9)
