
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 2)
        self.deconv = torch.nn.ConvTranspose2d(32, 32, kernel_size=2)

    def forward(self, x1):
        x = self.conv(x1)
        y = self.deconv(x) # Apply deconvolution on lastly-flooded tensor
        return y
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
