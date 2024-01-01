
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = torch.nn.GELU()
        self.conv_t1 = torch.nn.ConvTranspose1d(793, 4, 6, bias=False)
        self.conv_t2 = torch.nn.ConvTranspose1d(4, 7, 6, bias=False)
    def forward(self, x1, x18):
        x4 = self.conv_t1(x1)
        x5 = x4 > 0
        x6 = x4 * -0.07
        x7 = torch.where(x5, x4, x6)
        x8 = torch.nn.ReLU()(x7)
        x9 = torch.conv1d(x8, x18)
        x10 = x9 > 0
        x11 = x9 * -1.243
        x12 = torch.where(x10, x9, x11)
        x13 = self.conv_t2(x12)
        x14 = x13 > 0
        x15 = x13 * -1.042
        x16 = torch.where(x14, x13, x15)
        x17 = self.gelu(x16)
        return torch.conv1d(x17, x1)
# Inputs to the model
x1 = torch.randn(4, 793, 137)
x18 = torch.randn(7, 4, 30)
