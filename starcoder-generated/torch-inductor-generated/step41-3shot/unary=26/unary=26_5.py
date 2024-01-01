
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(54, 23, 4, stride=2, padding=1)
    def forward(self, x6):
        out = self.conv_t(x6)
        mask = out > 0
        mul = out * -0.7
        out = torch.where(mask, out, mul)
        out = torch.nn.functional.adaptive_max_pool2d(out, (1, 1))
        return torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
# Inputs to the model
x6 = torch.randn(13, 54, 14, 22)
