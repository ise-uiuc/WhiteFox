
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(6, 93, kernel_size=(6, 9), stride=(5, 5), padding=(1, 2), dilation=(2, 2))
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(38, 6, 62, 64)
