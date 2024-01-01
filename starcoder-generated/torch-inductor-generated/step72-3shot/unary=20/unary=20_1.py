
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(56, 56, kernel_size=(2, 3, 5), stride=(1, 2, 3), padding=(0, 1, 2), dilation=(1, 3, 4))
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 56, 15, 18, 20)
