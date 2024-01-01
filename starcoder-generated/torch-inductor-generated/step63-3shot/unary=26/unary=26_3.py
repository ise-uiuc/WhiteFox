
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(27, 14, kernel_size=(11, 95), padding=(5, 11))
    def forward(self, x28):
        v7 = self.conv_t(x28)
        v8 = v7 > 0
        v9 = v7 * 3.194
        v10 = torch.where(v8, v7, v9)
        return v10
# Inputs to the model
x28 = torch.randn(64, 27, 13, 56)
