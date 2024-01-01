
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=2, stride=2, padding=0)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=2, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose_2(x1)
        v2 = self.conv_transpose_1(v1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 2, 64, 64)
