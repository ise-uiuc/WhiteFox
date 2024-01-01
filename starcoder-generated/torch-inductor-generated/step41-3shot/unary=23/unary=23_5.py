
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, kernel_size=1, bias=True)
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 5, kernel_size=2, bias=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 32, 32)
