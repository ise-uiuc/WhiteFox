
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(113, 97, 41, stride=2, padding=1, bias=False)
    def forward(self, x):
        x1 = torch.nn.functional.adaptive_avg_pool2d(x, (18, 18))
        x2 = torch.transpose(x1, 2, 3)
        x3 = self.conv_t(x2)
        x4 = x3 > 0
        x5 = x3 * -55.278
        x6 = torch.where(x4, x3, x5)
        return torch.transpose(x6, 2, 3)
# Inputs to the model
x = torch.randn(1, 113, 31, 17)
