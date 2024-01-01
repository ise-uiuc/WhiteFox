
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose0 = torch.nn.ConvTranspose2d(1, 24, 1, stride=1, padding=0, output_padding=0)
        self.conv_transpose1 = torch.nn.ConvTranspose2d(24, 16, 1, stride=1, padding=0, output_padding=0)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(16, 3, 1, stride=1, padding=0, output_padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose0(x1)
        v2 = self.conv_transpose1(v1)
        v3 = self.conv_transpose2(v2)
        v4 = torch.sigmoid(v3)
        return v4

x1 = torch.randn(1, 1, 2, 2)
