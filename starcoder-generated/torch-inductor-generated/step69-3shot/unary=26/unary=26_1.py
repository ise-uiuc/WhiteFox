
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(64, 128, kernel_size=2, stride=3, padding=5, groups=7, bias=True)
    def forward(self, x):
        v1 = self.conv_t(x)
        v2 = v1 > 0
        v3 = v1 * 1.202
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x = torch.randn(6, 64, 52, 52)
