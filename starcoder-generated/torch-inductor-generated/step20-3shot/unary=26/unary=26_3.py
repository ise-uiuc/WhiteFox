
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(12, 6, 7, stride=3)
        self.conv_t2 = torch.nn.ConvTranspose2d(6, 1, 7, stride=3)
    def forward(self, x1):
        x2 = self.conv_t1(x1)
        x3 = x2 > 0
        x4 = x2 * 0.5
        x5 = torch.where(x3, x2, x4)
        x6 = self.conv_t2(x5)
        x7 = x6 > 0
        x8 = x6 * 0.5
        x9 = torch.where(x7, x6, x8)
        return x9
# Inputs to the model
x1 = torch.randn(16, 12, 64, 64)
