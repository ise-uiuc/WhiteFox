
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid1 = torch.nn.Sigmoid()
        self.sigmoid2 = torch.nn.Sigmoid()
        self.conv_t = torch.nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=(1, 1), padding=(1, 1), groups=2, bias=True)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.sigmoid1(v1)
        v3 = self.sigmoid2(v2)
        v4 = self.sigmoid1(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
