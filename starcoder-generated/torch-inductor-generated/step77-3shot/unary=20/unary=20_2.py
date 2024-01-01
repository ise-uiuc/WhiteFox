
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(13, 35, kernel_size=(1, 3, 3), stride=(1, 3, 3), padding=(0, 1, 1), dilation=1)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 13, 4, 3, 3)
