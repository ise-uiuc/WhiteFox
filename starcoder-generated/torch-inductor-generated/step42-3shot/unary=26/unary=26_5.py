
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(16, 192, 3, stride=2, padding=2, bias=False)
    def forward(self, x):
        v1 = self.conv_t(x)
        v2 = v1 > 0
        v3 = v1 * -0.481
        v4 = torch.where(v2, v1, v3)
        return torch.nn.functional.adaptive_avg_pool2d(v4, (1, 1))
# Inputs to the model
x = torch.randn(1, 16, 128, 25, 30)
