
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(288, 215, 6, stride=4, padding=3, output_padding=1, bias=False)
    def forward(self, x22):
        v1 = self.conv_t(x22)
        v2 = v1 > 0
        v3 = v1 * -0.016042025
        v4 = torch.where(v2, v1, v3)
        return torch.nn.functional.interpolate(v4, size=[196, 55])
# Inputs to the model
x22 = torch.randn(10, 288, 30, 34)
