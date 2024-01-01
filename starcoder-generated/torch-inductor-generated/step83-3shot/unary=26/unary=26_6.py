
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv_t = torch.nn.ConvTranspose2d(64, kernel_size=(1, 1), stride=(2, 2), activation="tanh")
    def forward(self, x1):
        x2 = torch.nn.LeakyReLU(1, True)(self.conv(x1))
        x3 = torch.nn.ELU(alpha=1.0000000000000000e+00, inplace=False)(x2)
        x4 = torch.nn.ReLU(inplace=False)(x3)
        x5 = torch.sigmoid(self.conv_t(x4))
        x6 = torch.abs(x5) + 1
        x7 = x6 * 0
        x8 = torch.tanh(x7) > 0
        x9 = x7 / 25
        x10 = torch.where(x8, x5, x9)
        return torch.abs(x10)
# Inputs to the model
x1 = torch.randn(6, 64, 75, 69)
