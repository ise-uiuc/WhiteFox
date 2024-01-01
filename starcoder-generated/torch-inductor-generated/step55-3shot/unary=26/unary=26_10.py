
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose3d(1, 4, 7, stride=2, bias=False)
        self.conv_t2 = torch.nn.ConvTranspose3d(4, 8, 6, stride=2, bias=False)
        self.conv_t3 = torch.nn.ConvTranspose3d(8, 6, 4, stride=2, bias=False)
    def forward(self, x3):
        x4 = self.conv_t1(x3)
        x5 = x4 > 0
        x6= x4 * 0.2909
        x7 = torch.where(x5, x4, x6)
        x8 = self.conv_t2(x7)
        x9 = x8 > 0
        x10 = x8 * 1.9325
        x11 = torch.where(x9, x8, x10)
        x12 = self.conv_t3(x11)
        x13 = x12 > 0
        x14 = x12 * 0.2912
        x15 = torch.where(x13, x12, x14)
        return x15
# Inputs to the model
x3 = torch.randn(12, 1, 20, 24, 24)
