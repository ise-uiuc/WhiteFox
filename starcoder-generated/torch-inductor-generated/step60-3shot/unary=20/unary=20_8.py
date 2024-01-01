
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=1, padding=0)
    def forward(self, x1):
        u1 = self.conv_t(x1)
        u2 = torch.sigmoid(u1)
        return u2
# Inputs to the model
x1 = torch.randn(1, 1024, 14, 14, 14)
