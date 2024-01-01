
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t_nchw = torch.nn.ConvTranspose2d(1, 2048, kernel_size=(1, 1))
    def forward(self, x1):
        v1 = self.conv_t_nchw(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 8192, 1)
