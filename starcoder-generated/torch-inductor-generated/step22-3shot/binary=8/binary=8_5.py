
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1)

    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        v2 = v1 + other

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
other = torch.randn(1, 8, 64, 64)
