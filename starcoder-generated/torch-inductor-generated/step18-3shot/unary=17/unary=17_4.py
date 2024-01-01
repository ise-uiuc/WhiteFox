
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), output_padding=(1, 1), dilation=1, groups=1)
    def forward(self, x_in):
        v1 = self.conv_transpose(x_in)
        v1 = torch.relu(v1)
        return v1
# Inputs to the model
x_in = torch.randn(1, 16, 160, 160)
