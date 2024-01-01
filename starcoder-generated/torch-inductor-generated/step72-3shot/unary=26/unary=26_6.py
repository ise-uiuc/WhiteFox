
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(783, 2, 4, stride=4)
    def forward(self, x0):
        a1 = self.conv_t(x0)
        a2 = a1 > 0
        a3 = a1 * -0.4070818573854245
        a4 = torch.where(a2, a1, a3)
        a5 = a4 * 0.4043118710124682
        a6 = a1 * -0.5228572773901759
        return a5
# Inputs to the model
x0 = torch.randn(3, 783, 251, 133)
