
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, out_channels=4, kernel_size=1, stride=(1, 1), bias=True)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
