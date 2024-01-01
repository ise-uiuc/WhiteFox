
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 5, 3, stride=2, padding=4, output_padding=2)
    def forward(self, x0):
        v1 = self.conv_t(x0)
        v2 = v1 > 0
        v3 = v1 * 0.43
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x0 = torch.randn(17, 1, 16, 16)
