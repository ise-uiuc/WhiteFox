
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(24, 48, kernel_size=(2, 2), padding=(1, 1), bias=False)
        self.conv_t2 = torch.nn.ConvTranspose2d(48, 32, kernel_size=(2, 2), padding=(1, 1), bias=False)
    def forward(self, x1):
        v1 = self.conv_t1(x1)
        v2 = self.conv_t2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 24, 32, 32)
