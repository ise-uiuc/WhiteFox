
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(2, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, output_padding=(0, 0), bias=True, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 96, 96)
