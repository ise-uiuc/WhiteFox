
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(35, 35, 11, stride=1, padding=5, bias=False)
    def forward(self, x1):
        x1 = self.conv_t(x1)
        x2 = x1 > 0
        x3 = x1 * 100
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x1 = torch.randn(4, 35, 13, 76)
