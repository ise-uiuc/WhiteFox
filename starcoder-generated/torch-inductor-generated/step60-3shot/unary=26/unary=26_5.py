
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = nn.ConvTranspose2d(64, 64, 4, bias=False, stride=2, padding=1)
    def forward(self, x):
        x1 = self.conv_t(x)
        x2 = x1 > 0
        x3 = x1 * -0.79
        x4 = torch.where(x2, x1, x3)
        return torch.nn.functional.adaptive_avg_pool2d(torch.nn.ReLU()(x4), (1, 1))
# Inputs to the model
x = torch.randn(1, 64, 1080, 1280)
