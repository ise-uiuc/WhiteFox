
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(30, 39, 7, stride=1, bias=True)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * 0.24
        x4 = torch.where(x2, x1, x3)
        return x4.abs()
# Inputs to the model
x = torch.randn(1, 30, 43, 12)
