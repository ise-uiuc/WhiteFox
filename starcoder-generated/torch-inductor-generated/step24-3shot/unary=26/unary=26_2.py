
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 9, 7, stride=5, dilation=2)
    def forward(self, x2):
        v1 = self.conv_t(x2)
        v2 = v1 >= 0
        v3 = v1 * 0.103
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x2 = torch.randn(9, 3, 14, 11)
