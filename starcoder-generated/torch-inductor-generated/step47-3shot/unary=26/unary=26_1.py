
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(306, 254, 8, stride=1, padding=0, bias=True)
    def forward(self, x3):
        j1 = self.conv_t(x3)
        j2 = j1 > 0
        j3 = j1 * -0.909
        j4 = torch.where(j2, j1, j3)
        return torch.nn.functional.adaptive_avg_pool2d(j4, (1, 1))
# Inputs to the model
x3 = torch.randn(88, 306, 12, 20)
