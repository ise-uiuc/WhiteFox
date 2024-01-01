
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(256, 256, kernel_size=1)
    def forward(self, x):
        t1 = self.conv(x)
        t2 = t1 + 3
        t3 = t2.clamp(0, 6)
        t4 = t1 * t3
        t5 = t4 / 6
        t6 = t5 / 6
        return t6
# Inputs to the model
x = torch.randn(1, 256, 224, 224)
