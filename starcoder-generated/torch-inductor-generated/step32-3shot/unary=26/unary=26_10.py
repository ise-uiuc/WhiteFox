
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 64, 4, stride=2, padding=3, bias=True)
    def forward(self, input5):
        v1 = self.conv_t(input5)
        v2 = v1 > 0
        v3 = v1 * -3.83
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
input5 = torch.randn(1, 3, 17, 15)
