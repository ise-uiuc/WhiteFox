
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose1d(in_channels=3, out_channels=32, kernel_size=(1,), stride=(4,), groups=32, bias=True, padding=0, dilation=1)
        self.gelu = torch.nn.GELU()
        self.conv_t2 = torch.nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=(4,), stride=(1,), groups=32, bias=True, padding=0, dilation=1)
        self.conv_t3 = torch.nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=(3,), stride=(4,), groups=32, bias=True, padding=0, dilation=1)
        self.conv_t4 = torch.nn.ConvTranspose1d(in_channels=32, out_channels=20, kernel_size=(1,), stride=(4,), groups=20, bias=True, padding=0, dilation=1)
    def forward(self, x8):
        h1 = self.conv_t1(x8)
        h2 = self.gelu(h1)
        h3 = self.conv_t2(h2)
        h4 = self.gelu(h3)
        h5 = self.conv_t3(h4)
        h6 = self.gelu(h5)
        h7 = self.conv_t4(h6)
        return h7
# Inputs to the model
x8 = torch.randn(1, 3, 32)
