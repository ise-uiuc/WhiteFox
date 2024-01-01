
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 1, kernel_size=2, stride=2)
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=2, stride=2)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv(v1)
        return v2
# Inputs to the model
x1 = torch.randn(3, 3, 3, 3)
