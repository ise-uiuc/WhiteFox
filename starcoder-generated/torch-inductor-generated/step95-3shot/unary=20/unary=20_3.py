
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 2, kernel_size=3)
        self.conv_t1 = torch.nn.ConvTranspose2d(2, 2, kernel_size=3)
    def forward(self, x1):
        x2 = self.conv_t(x1)
        x3 = self.conv_t1(x2)
        v2 = torch.sigmoid(x3)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 336, 384)
