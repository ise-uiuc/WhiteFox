
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 5, stride=2)
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, 8, stride=1, padding=1)
    def forward(self, x4):
        v1 = self.conv(x4)
        v2 = self.conv_t(v1)
        v3 = v2 > 0
        v4 = v2 * 0.130
        v5 = torch.where(v3, v2, v4)
        return v5
# Inputs to the model
x4 = torch.randn(1, 3, 256, 256)
