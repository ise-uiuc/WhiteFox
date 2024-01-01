
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(198, 74, 3, stride=2, padding=0, output_padding=0, bias=False)
    def forward(self, x18):
        v1 = self.conv_t(x18)
        v2 = v1 > 1
        v3 = v1 * -0.866664
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x18 = torch.randn(15, 198, 56, 68)
