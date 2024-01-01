
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 1, kernel_size=3, bias=False, stride=2)
    def forward(self, y01):
        y02 = self.conv_t(y01)
        y03 = self.conv_t.weight >= 0
        y04 = y02 * -0.1597
        y05 = torch.where(y03, y02, y04)
        y06 = torch.clamp(y05, 0.0, 5.0)
        y07 = torch.floor(y06)
        return y07
# Inputs to the model
y01 = torch.randn(4, 4, 8, 13)
