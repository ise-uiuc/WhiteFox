
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, out_channels=1, kernel_size=3, bias=False)
    def forward(self, x0):
        v0 = torch.permute(x0, [0, 2, 1])
        v0 = torch.permute(v0, [0, 2, 1])
        v1 = self.conv_t(v0)
        v2 = torch.sigmoid(v1)
        v3 = torch.inverse(v2)
        v4 = torch.view(v3, [1, 128])
        return v4
# Inputs to the model
x0 = torch.randn(1, 1, 64, 64)
