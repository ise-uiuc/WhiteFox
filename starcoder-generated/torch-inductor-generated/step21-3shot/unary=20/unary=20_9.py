
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.conv_t = torch.nn.ConvTranspose2d(in_channels=10, kernel_size=3)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(5, 10, 10, 10)
