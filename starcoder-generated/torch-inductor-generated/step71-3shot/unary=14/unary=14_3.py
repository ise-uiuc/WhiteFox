
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=44, groups=3, bias=False, stride=(2,1), padding=(1,1))
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(3, 3, kernel_size=(3, 3), groups=1, bias=False, stride=(2,1), padding=(4,1))
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.conv_transpose_1(v1)
        v3 = torch.sigmoid(v2)
        v4 = v2 * v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
