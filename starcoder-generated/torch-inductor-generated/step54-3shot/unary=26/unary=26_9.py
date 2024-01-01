
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(3, 6, 3, stride=2, padding=1, bias=False)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * -3
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x = torch.randn(1, 3, 17, 23, 33)
