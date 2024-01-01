
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(19, 26, (3, 4), stride=(2, 2), padding=(2, 0), dilation=(1, 1),groups=1, bias=False)
    def forward(self, x):
        v1 = self.conv_t(x)
        v2 = v1 > 0
        v3 = v1 * 4.4892
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x = torch.randn(1, 19, 12, 22)
