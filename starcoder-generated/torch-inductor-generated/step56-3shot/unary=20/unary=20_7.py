
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(12, 12, kernel_size=10, stride=10)
        self.conv = torch.nn.Conv2d(12, 12, kernel_size=3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.conv(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 12, 64, 64)
