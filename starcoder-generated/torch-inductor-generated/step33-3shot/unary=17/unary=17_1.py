
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c4 = torch.nn.ConvTranspose2d(128, 64, kernel_size=1)
        self.c8 = torch.nn.ConvTranspose2d(64, 32, kernel_size=1)
        self.c16 = torch.nn.ConvTranspose2d(32, 4, kernel_size=1)
        self.c32 = torch.nn.ConvTranspose2d(4, 1, kernel_size=1)
    def forward(self, x):
        x = self.c4(x)
        x = self.c8(x)
        x = self.c16(x)
        x = self.c32(x)
        return x
# Model inputs of type float
x1 = torch.randn(1, 128, 2, 2)
