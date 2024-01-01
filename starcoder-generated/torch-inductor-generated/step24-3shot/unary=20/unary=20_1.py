
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(1, 3, kernel_size=5, dilation=1)
    def forward(self, x1):
        x1 = self.conv_t(x1)
        x1 = torch.sigmoid(x1)
        return x1
# Inputs to the model
x1 = torch.randn(4, 1, 1, 41, 275)
