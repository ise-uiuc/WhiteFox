
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv = torch.nn.Conv2d(39, 1, (1, 7), stride=(5, 2))
        self.conv_t_neg = torch.nn.ConvTranspose2d(1, 1, (1, 6), stride=(5, 1))
        self.conv_t_pos = torch.nn.ConvTranspose2d(1, 1, (1, 7), stride=(1, 7))
        self.negative_slope = negative_slope
    def forward(self, x):
        y = self.conv(x)
        z1 = self.conv_t_neg(y)
        z2 = self.conv_t_pos(z1)
        m1 = z2 > 0
        m2 = z2 * self.negative_slope
        m3 = torch.where(m1, z2, m2)
        return m3
negative_slope = 0.6619
# Inputs to the model
x = torch.randn(1, 39, 54, 56)
