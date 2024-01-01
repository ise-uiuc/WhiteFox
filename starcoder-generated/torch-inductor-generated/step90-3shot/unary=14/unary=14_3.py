
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_100 = torch.nn.ConvTranspose2d(18, 18, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0))
    def forward(self, x1):
        v1 = self.conv_transpose_100(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 18, 16, 16)
