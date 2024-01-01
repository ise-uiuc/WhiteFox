
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(32, 64, kernel_size=(
            1, 2), stride=(3, 2), padding=(1, 0), dilation=(2, 1), output_padding=(0, 1))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 32, 12, 10)
