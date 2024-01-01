
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=(2, 3, 4), stride=(1, 2, 1), padding=(1, 2, 1), output_padding=(1, 2, 2))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 65, 10, 34)
