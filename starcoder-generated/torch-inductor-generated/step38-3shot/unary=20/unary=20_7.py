
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(4, 6, kernel_size=(2, 6), stride=(3, 1), padding=(1, 2), dilation=(2, 2))
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 3, 2)
