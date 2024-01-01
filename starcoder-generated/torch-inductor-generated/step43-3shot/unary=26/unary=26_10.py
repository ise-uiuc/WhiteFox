
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(64, 984, 15, stride=1, padding=0, bias=False)
    def forward(self, x2):
        f6 = self.conv_t(x2)
        f5 = f6 > 0
        f8 = f6 * -0.2458
        f3 = torch.where(f5, f6, f8)
        return f3
# Inputs to the model
x2 = torch.randn(41, 64, 74, 87)
