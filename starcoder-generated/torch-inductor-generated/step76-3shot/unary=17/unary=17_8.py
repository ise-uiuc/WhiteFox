
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose1d(512, 512, kernel_size=2, padding=1, stride=1, dilation=1, output_padding=0, groups=1, bias=True)
        self.conv_transpose_2 = torch.nn.ConvTranspose1d(512, 128, kernel_size=2, padding=1, stride=2, dilation=1, output_padding=0, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = torch.relu(v1)
        v3 = torch.tanh(v2)
        v4 = self.conv_transpose_2(v3)
        v5 = torch.relu(v4)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 512)
