
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(129, 25, 7, stride=1, padding=0, output_padding=1, bias=False)
    def forward(self, x2):
        v1 = self.conv_t(x2)
        v2 = v1 > 0
        v3 = v1 * 0.622
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x2 = torch.randn(32, 129, 83, 35)
