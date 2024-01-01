
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 62, (3, 1), stride=(3, 3), padding=(1, 0), output_padding=(1, 0), dilation=(2, 4), groups=(1, 1))
    def forward(self, x6):
        v1 = self.conv_t(x6)
        v2 = v1 > 0
        v3 = v1 * 0.1
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x6 = torch.randn(1, 1, 5, 5)
