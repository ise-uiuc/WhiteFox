
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(128, 512, kernel_size=2, stride=(1, 2), padding=(0, 1), output_padding=(0, 1), groups=1, bias=True, dilation=1, padding_mode='zeros')
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
        self.conv_transpose_4 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, stride=3, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
        #self.conv_transpose_5 = torch.nn.ConvTranspose2d(64, 32, kernel_size=7, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = self.conv_transpose_2(v1)
        v3 = self.conv_transpose_3(v2)
        v4 = self.conv_transpose_4(v3)
        v5 = self.conv_transpose_5(v4)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 32, 210, 9)
