
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dconv_6_1 = torch.nn.ConvTranspose2d(24, 512, kernel_size=4, stride=2, padding=1, groups=4, bias=False)
    def forward(self, x1):
        v1 = self.dconv_6_1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 24, 8, 8)
