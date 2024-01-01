
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(88, 28, 3, bias=False)
    def forward(self, x13):
        x14 = self.conv_t(x13)
        x15 = x14 > 0
        x16 = x14 * -0.2103
        x17 = torch.where(x15, x14, x16)
        return torch.nn.functional.avg_pool2d(x17, 3)
# Inputs to the model
x13 = torch.randn(68, 88, 8, 73)
