
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose0 = torch.nn.ConvTranspose2d(40, 40, kernel_size=3, stride=2, padding=1, dilation=2, groups=1, bias=True)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(in_channels=10, out_channels=10, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=2, groups=1, bias=True)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(in_channels=20, out_channels=20, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=2, groups=1, bias=True)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(in_channels=10, out_channels=10, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=2, groups=1, bias=True)
        self.conv_transpose4 = torch.nn.ConvTranspose2d(in_channels=30, out_channels=30, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=2, groups=1, bias=True)
        self.conv_transpose5 = torch.nn.ConvTranspose2d(in_channels=10, out_channels=10, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=2, groups=1, bias=True)
        self.conv_transpose6 = torch.nn.ConvTranspose2d(in_channels=40, out_channels=40, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=2, groups=1, bias=True)
        self.conv_transpose7 = torch.nn.ConvTranspose2d(in_channels=10, out_channels=10, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=2, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose0(x1)
        v2 = self.conv_transpose1(v1)
        v3 = self.conv_transpose2(v2)
        v4 = self.conv_transpose3(v3)
        v5 = self.conv_transpose4(v4)
        v6 = self.conv_transpose5(v5)
        v7 = self.conv_transpose6(v6)
        v8 = self.conv_transpose7(v7)
        v9 = torch.sigmoid(v8)
        return v9
x1 = torch.randn(1, 40, 43, 72)
