
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(530, 371, 6, stride=1, padding=1, bias=False)
    def forward(self, x8):
        v1 = self.conv_t(x8)
        v2 = v1 > 0
        v3 = v1 * -0.28
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x8 = torch.randn(2, 530, 54, 73)
