
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(64, 256, 3)
        self.conv_transpose_5 = torch.nn.ConvTranspose2d(256, 64, 1)
        self.conv_transpose_6 = torch.nn.ConvTranspose2d(64, 1, 1)
    def forward(self, x):
        conv_out = self.conv(x)
        relu_1 = torch.relu(conv_out)
        transpose_5 = self.conv_transpose_5(relu_1)
        relu_2 = torch.relu(transpose_5)
        transpose_6 = self.conv_transpose_6(relu_2)
        sigmoid = torch.sigmoid(transpose_6)
        return sigmoid
# Inputs to the model
x = torch.randn(1, 64, 64, 64)
