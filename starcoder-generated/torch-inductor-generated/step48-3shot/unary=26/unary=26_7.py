
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(9, 5, 2, stride=1, padding=1, bias=False)
    def forward(self, x4):
        v1 = self.conv_t(x4)
        v2 = v1 > 0
        v3 = v1 * -1
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x4 = torch.randn(1, 9, 5, 2)
