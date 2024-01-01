
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(32, 1024, kernel_size=3, stride=2, padding=1)
        self.conv_t = torch.nn.ConvTranspose3d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.conv_t = torch.nn.ConvTranspose3d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv_t = torch.nn.ConvTranspose3d(512, 256, kernel_size=3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
