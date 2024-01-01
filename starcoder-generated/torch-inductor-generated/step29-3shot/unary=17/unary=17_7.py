
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(12, 64, kernel_size=7, stride=(2, 1))
        self.conv1 = torch.nn.ConvTranspose2d(64, 256, kernel_size=7, stride=(2, 1))
        self.conv2 = torch.nn.ConvTranspose2d(256, 8, kernel_size=7, stride=(1, 1))

    def forward(self, x1):
        y_7 = self.conv(x1)
        y_8 = self.conv1(y_7)
        y_9 = self.conv2(y_8)
        return y_9
# Inputs to the model
x1 = torch.randn(1, 12, 7, 7)
