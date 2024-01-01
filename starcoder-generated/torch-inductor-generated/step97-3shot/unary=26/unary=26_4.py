
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(76, 64, 1, stride=4, padding=3, bias=False)
    def forward(self, x3):
        x1 = self.conv_t(x3)
        x2 = x1 > 0
        x3 = x1 * -0.412
        x4 = torch.where(x2, x1, x3)
        return torch.transpose(torch.nn.functional.adaptive_avg_pool2d(torch.nn.Softplus()(x4), (1, 6)), 2, 3)
# Inputs to the model
x3 = torch.randn(1, 76, 30, 59)
