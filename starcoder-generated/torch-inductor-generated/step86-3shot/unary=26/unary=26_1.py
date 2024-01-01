
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(267, 148, 1, stride=1, padding=0, output_padding=4, bias=False)
    def forward(self, x0):
        v1 = self.conv_t(x0)
        v2 = v1 > 0
        v3 = v1 * -3.35
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x0 = torch.randn(48, 267, 8, 5)
