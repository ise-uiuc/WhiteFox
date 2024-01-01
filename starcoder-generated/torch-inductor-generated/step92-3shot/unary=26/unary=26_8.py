
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(8, 12, 5, stride=1, padding=2, bias=False)
    def forward(self, x15):
        x1 = self.conv_t(x15)
        x2 = x1 > 0
        x3 = x1 * -4.87873
        x4 = torch.where(x2, x1, x3)
        return x4
# Inputs to the model
x15 = torch.randn(60, 8, 373, 51)
# Inputs to the model
x18 = torch.randn(60, 8, 373, 51)
