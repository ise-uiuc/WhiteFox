
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_x = torch.nn.ConvTranspose2d(in_channels=160, out_channels=96, kernel_size=1, stride=1, padding=1, output_padding=1)
        self.conv_transpose_y = torch.nn.ConvTranspose2d(in_channels=160, out_channels=80, kernel_size=1, stride=1, padding=1, output_padding=1)
    def forward(self, x):
        x1 = self.conv_transpose_x(x)
        x2 = torch.relu(x1)
        x3 = self.conv_transpose_y(x)
        x4 = torch.relu(x3)
        return x4, x2
x1 = torch.randn(1, 160, 10, 10)
m = Model()
m(x1)
