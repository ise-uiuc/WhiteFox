
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, kernel_size=3)
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 1, kernel_size=3)
    def forward(self, x):
        x = x.view(1, 1, 28, 28)
        x = self.conv(x)
        x = self.conv_transpose(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
