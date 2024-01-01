
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(234, 196, 6, stride=2, padding=2, output_padding=1, bias=False)
    def forward(self, x21):
        v1 = self.conv_t(x21)
        v2 = v1 > 0
        v3 = v1 * 0.295
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x21 = torch.randn(10, 234, 30, 15)
