
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 6, kernel_size=(7, 7), padding=(3, 3))
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 0x1.e-1d2cfep9
        return v2
# Inputs to the model
x = torch.randn(1, 1, 512, 512)
