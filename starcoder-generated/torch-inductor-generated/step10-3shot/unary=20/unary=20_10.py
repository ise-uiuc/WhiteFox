
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 32, 60, 60)
