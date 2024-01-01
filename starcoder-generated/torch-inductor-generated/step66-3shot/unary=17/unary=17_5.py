
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.ConvTranspose2d(1, 128, 5, 3, 2)
        self.conv_transpose2d = torch.nn.ConvTranspose2d(128, 64, 3, 2, 1)
        self.conv2d1 = torch.nn.ConvTranspose2d(64, 32, 3, 2, 1)
        self.conv2d2 = torch.nn.ConvTranspose2d(32, 16, 3, 1, 1)
        self.conv2d3 = torch.nn.ConvTranspose2d(16, 1, 3, 1, 2)
        self.activation = torch.nn.ReLU()
        self.pooling1 = torch.nn.MaxPool2d(3, 2, 1)
        self.maxpool = torch.nn.MaxPool2d(3, 2, 1)
        self.max_pool3d = torch.nn.MaxPool3d(1, 1, 1)
    def forward(self, x):
        x0 = self.conv2d(x)
        x1 = self.conv_transpose2d(x0)
        x2 = self.activation(x1)
        x3 = self.pooling1(x2)
        x4 = self.conv2d1(x3)
        x5 = self.activation(x4)
        x6 = self.pooling1(x5)
        x7 = self.conv2d2(x6)
        x8 = self.activation(x7)
        x9 = self.pooling1(x8)
        x10 = self.conv2d3(x9)
        x11 = self.max_pool3d(x10)
        x12 = self.maxpool(x11)
        output = torch.tanh(x12)
        return output
# Inputs to the network
x = torch.randn(1, 1, 320, 380)
