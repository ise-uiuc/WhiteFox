
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = torch.nn.ConvTranspose2d(16, 240, kernel_size=3, stride=5, padding=1, output_padding=(1, 0))
    def forward(self, x1):
        v1 = self.deconv(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32)
