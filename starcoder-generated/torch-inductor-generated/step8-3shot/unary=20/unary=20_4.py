
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(2, 2, kernel_size=(2, 2, 7), stride=(2, 2, 3), padding=(1, 1, 4),
                                                dilation=(1, 1, 3), groups=1, bias=True, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return x2v2
# Inputs to the model
x1 = torch.randn(1, 2, 15, 13, 10)
