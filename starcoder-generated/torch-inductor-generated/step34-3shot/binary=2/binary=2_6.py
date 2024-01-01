
class Model(torch.nn.Module):
    def __init__(self):
        super().init()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1, 2), stride=(1, 2), padding=(2, 3))
    def forward(self, x):
        x = self.conv(x)
        return x - (0.25).type(torch.double)
# Inputs to the model
x = torch.randn(5, 3, 16, 32)
