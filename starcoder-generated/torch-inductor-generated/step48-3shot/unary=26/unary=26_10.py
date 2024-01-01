
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(15, 6, 4,stride=2, padding=2, bias=True)
    def forward(self, x11):
        out = self.conv_t(x11)
        mask = out > 0
        mul = out * -0.82
        out = torch.where(mask, out, mul)
        out = torch.nn.functional.adaptive_max_pool2d(out, (1, 1))
        return torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
# Inputs to the model
x11 = torch.randn(12, 15, 7, 8)
