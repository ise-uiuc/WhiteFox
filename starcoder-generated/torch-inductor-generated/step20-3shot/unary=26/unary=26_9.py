
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(672, 232, 1, stride=1)
        self.conv_t_1 = torch.nn.ConvTranspose2d(672, 232, 1, stride=1)
        self.conv_t_2 = torch.nn.ConvTranspose2d(768, 768, 1, stride=2)
        self.conv_t_3 = torch.nn.ConvTranspose2d(768, 768, 1, stride=1)
    def forward(self, x1, x2):
        x3 = self.conv_t(x1)
        x4 = self.conv_t_1(x2)
        x5 = torch.cat([x3, x4], dim=1)
        x6 = self.conv_t_2(x5)
        x7 = x6 <= 0
        x8 = x6 * 0.5
        x9 = torch.where(x7, x6, x8)
        x10 = self.conv_t_3(x9)
        return x10
# Inputs to the model
x1 = torch.randn(32, 672, 56, 56)
x2 = torch.randn(32, 768, 28, 28)
