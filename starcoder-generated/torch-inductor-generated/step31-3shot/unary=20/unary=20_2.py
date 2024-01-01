
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.conv2d = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_transpose_3 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = self.softmax(x1)
        v2 = self.conv2d(v1)
        v3 = self.conv_transpose_1(v2)
        v4 = self.conv_transpose_2(v3)
        v5 = self.conv_transpose_3(v4)
        v6 = self.softmax(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 108, 108)
