
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(10, stride=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(64, 32, kernel_size=10, padding=10, stride=10, dilation=10, groups=32)
    def forward(self, x1):
        v1 = self.pool(x1)
        v2 = self.conv_transpose(v1)
        v3 = v2.reshape(-1, 32, 100)
        return (v3, v1)
# Inputs to the model
x1 = torch.randn(1, 64, 500, 600)
