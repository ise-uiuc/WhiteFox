
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d( 1, 1, kernel_size=34, stride=1, padding=7, output_padding=33, dilation=3)
        self.conv_t2 = torch.nn.ConvTranspose2d( 1, 1, kernel_size=13, stride=1, padding=1, output_padding=15, dilation=1)
        self.conv_t3 = torch.nn.ConvTranspose2d( 1, 1, kernel_size=11, stride=1, padding=7, output_padding=19, dilation=1)
    def forward(self, x1):
        v1 = self.conv_t1(x1)
        v2 = self.conv_t2(v1)
        v3 = self.conv_t3(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)
