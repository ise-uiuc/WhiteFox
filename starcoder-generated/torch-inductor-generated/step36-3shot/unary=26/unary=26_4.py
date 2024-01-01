
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(386, 139, 5, stride=1, padding=0, bias=False)
    def forward(self, x4):
        v1 = self.conv_t(x4)
        v2 = v1 > 0
        v3 = v1 * -1.5003
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x4 = torch.randn(13, 386, 21, 14)
