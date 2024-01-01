
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
        self.conv_transpose = torch.nn.ConvTranspose2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv_transpose(v4)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 96, 96, 192)
