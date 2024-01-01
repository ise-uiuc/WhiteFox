
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(22, 600, 3, stride=2, padding=14)
    def forward(self, x14):
        v1 = self.conv_t(x14)
        v2 = v1 > 0
        v3 = v1 * -0.036066667
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x14 = torch.randn(1, 22, 7, 7)
