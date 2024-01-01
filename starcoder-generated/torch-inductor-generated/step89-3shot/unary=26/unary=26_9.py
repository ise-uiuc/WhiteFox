
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(52, 3, kernel_size=(5, 6), stride=(5, 8), padding=(5, 10))

    def forward(self, x5):
        f1 = self.conv_t(x5)
        f2 = f1 > 0
        f3 = f1 * 0.1
        f4 = torch.where(f2, f1, f3)
        return f4
# Inputs to the model
x5 = torch.randn(54, 52, 13, 17)
