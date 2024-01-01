
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 64, kernel_size=13, stride=-14, bias=False)
    def forward(self, x1):
        x1 = torch.transpose(x1, 1, 3)
        x2 = torch.transpose(x1, 2, 3)
        x3 = self.conv_t(x2)
        x4 = torch.transpose(x3, 1, 3)
        x5 = torch.transpose(x4, 2, 3)
        x6 = x5 > 0
        x7 = x4 * 0.5
        x8 = torch.where(x6, x4, x7)
        return torch.abs(x8)
# Inputs to the model
x1 = torch.randn(3, 64, 12, 16)
