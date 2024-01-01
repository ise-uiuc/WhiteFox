
class Model(torch.nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, kernel_size=kernel_size)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1 > 0
        v3 = v1 * 0.1
        v4 = torch.where(v2, v1, v3)
        return v4
kernel_size = 2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
