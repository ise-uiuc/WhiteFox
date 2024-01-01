
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(11, 111, kernel_size=(1, 4), stride=(1, 1), padding=0, dilation=1)
    def forward(self, x5):
        v1 = self.conv_t(x5)
        v2 = v1 > 0.0222
        v3 = v1 * 0.9765
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x5 = x5 = torch.randn(19, 11, 3, 1)
