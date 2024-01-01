
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(64, 129, kernel_size=[7, 1], stride=[4, 1], padding=[2, 0], groups=8)
    def forward(self, x7):
        v1 = self.conv_t(x7)
        v2 = v1 > 0
        v3 = v1 * -1.27
        v4 = torch.where(v2, v1, v3)
        return torch.nn.functional.adaptive_max_pool2d(v4, (68, 22))
# Inputs to the model
x7 = torch.randn(19, 64, 28, 8)
