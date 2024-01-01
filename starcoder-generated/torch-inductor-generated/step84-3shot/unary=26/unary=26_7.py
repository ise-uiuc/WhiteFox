
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(122, 24, 1, bias=True)
        self.conv_t = torch.nn.ConvTranspose2d(24, 122, 21, stride=4, padding=20, output_padding=7, groups=10)
    def forward(self, x26):
        u1 = self.conv2d(x26)
        u2 = F.relu(u1)
        u3 = u2 > 0
        u4 = u2 * 42.0564
        u5 = torch.where(u3, u2, u4)
        u6 = torch.nn.functional.relu6(u5)
        u7 = self.conv_t(u6)
        u8 = (u7 > 0)
        u9 = (u7 * -0.08)
        u10 = torch.where(u8, u7, u9)
        return torch.nn.functional.relu6(u10)
# Inputs to the model
x26 = torch.randn(4, 122, 86, 41)
