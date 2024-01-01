
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(8, 16, 3, stride=2, dilation=1, padding=1)
        self.conv_transpose_relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv_transpose_relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 8, 128)
