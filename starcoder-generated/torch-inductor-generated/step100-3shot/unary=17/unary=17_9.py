
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 64, kernel_size=3)
        self.pool = torch.nn.MaxPool2d(kernel_size=3)
        self.convt = torch.nn.ConvTranspose2d(64, 1, kernel_size=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.pool(v1)
        v3 = self.convt(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
