
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        kernel_size = 2
        stride = 2
        padding = 1
        self.t_conv = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=kernel_size,
                                         stride=stride, padding=padding)
        self.sig = nn.Sigmoid()

    def forward(self, x1):
        x1 = self.t_conv(x1)
        x2 = self.sig(x1)

        return x2

model = MyModel()
# Inputs to the model
x1 = torch.randn(1, 8, 128, 128)
