
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(309, 439, 6, stride=3, padding=2, bias=False)
    def forward(self, x5):
        v1 = self.conv_t(x5)
        v2 = v1 > 0
        v3 = v1 * -1.0480081
        v4 = torch.where(v2, v1, v3)
        return v4 + torch.nn.functional.adaptive_avg_pool2d(v4, (1, 1))
# Inputs to the model
x5 = torch.randn(24, 309, 1, 5)
