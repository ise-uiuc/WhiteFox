
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(117, 156, 6, bias=False)
    def forward(self, x):
        x23 = self.conv_t(x)
        x24 = x23 > 0
        x25 = x23 * -0.4408
        x26 = torch.where(x24, x23, x25)
        return torch.nn.functional.adaptive_avg_pool2d(torch.nn.LeakyReLU(0.9104)(x26), (6, 0))
# Inputs to the model
x = torch.randn(4, 117, 15, 69, device='cuda')
