
class LeakySoftmaxModel(torch.nn.Module):
    def __init__(self, conv_transpose_in_channels):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(30, 50, 5)
        self.conv2 = torch.nn.ConvTranspose2d(10, 20, 8)
        self.conv3 = torch.nn.ConvTranspose2d(20, 24, 3, 1, 1)
        self.conv4 = torch.nn.ConvTranspose2d(10, 46, 5)
        self.convt = torch.nn.ConvTranspose2d(50, conv_transpose_in_channels, 1)
    def forward(self, input):
        x1 = self.conv2(self.conv1(input))
        x2 = torch.nn.LeakyReLU(0.05)(x1)
        x3 = torch.nn.functional.softmax(x2, dim=1)
        x4 = x1 + x3
        x5 = self.conv4(self.conv3(x4))
        x6 = torch.nn.LeakyReLU(0.02)(x5)
        x7 = torch.nn.functional.softmax(x6, dim=1)
        x8 = self.convt(x5 * x7)
        return x8
# Inputs to the model
input = torch.randn(1, 30, 57, 85)

# Model begins
class LeakySoftmaxModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(33, 55, 4, 2, padding=0, bias=False)
        self.conv2 = torch.nn.ConvTranspose2d(30, 36, 8, 1, bias=False)
        self.conv3 = torch.nn.ConvTranspose2d(36, 45, 3, 1, bias=False)
        self.conv4 = torch.nn.ConvTranspose2d(12, 48, 6, 1, bias=False)
    def forward(self, input):
        x1 = self.conv1(input)
        x2 = torch.nn.LeakyReLU(0.09)(x1)
        x3 = torch.nn.functional.softmax(x2, dim=1)
        x4 = x1 + x3
        x5 = self.conv4(self.conv3(self.conv2(x4)))
        x6 = torch.nn.functional.leaky_relu(x5)
        x7 = torch.nn.functional.softmax(x6, dim=1)
        return x5
# Inputs to the model
input = torch.randn(1, 33, 57, 85)
