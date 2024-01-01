
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(1, 1, 3)
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, kernel_size=3)
    def forward(self, x1):
        x = self.conv(x1) # convolution over 1x1 kernel
        x = self.relu(x) # applied to input x1
        x = self.conv_transpose(x)
        return x
# Inputs to the model
x1 = torch.randn(3, 1, 16, 16)
