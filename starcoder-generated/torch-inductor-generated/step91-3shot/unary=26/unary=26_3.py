
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose3d(99, 31, kernel_size=(3, 3, 6), stride=(3, 1, 6), padding=(1, 0, 1), output_padding=(0, 1, 5), groups=27, bias=False)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = v1 > 0
        v3 = v1 * -0.2277873
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(28, 99, 28, 14, 39)
