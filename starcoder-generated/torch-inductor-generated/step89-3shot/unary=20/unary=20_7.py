
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 10, kernel_size=3, stride=1,padding=1, bias=True)
        self.conv_t = torch.nn.ConvTranspose2d(10, 10, kernel_size=3, stride=1, padding=1,bias=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_t(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 10, 45, 45)
