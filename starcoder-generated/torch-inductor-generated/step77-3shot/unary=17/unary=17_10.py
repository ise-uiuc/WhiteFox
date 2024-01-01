
class conv_transpose(torch.nn.Module):
    def __init__(self):
        super(conv_transpose, self).__init__()
        self.conv2d = torch.nn.ConvTranspose2d(in_channels=1, out_channels=3, kernel_size=3, padding=1, output_padding=1, stride=1)
    def forward(self, x1):
        r = self.conv2d(x1)
        return r

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_transpose = conv_transpose()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 4, 4)
